from typing import List

import torch
import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def pad_tensor(tensor, size):
    padded = torch.zeros(size, dtype=tensor.dtype, device=tensor.device)
    slices = tuple(slice(0, min(s, ps)) for s, ps in zip(tensor.shape, size))
    padded[slices] = tensor[slices]
    return padded


def all_gather_tensor_shapes(tensor):
    world_size = dist.get_world_size()
    all_shapes = [None for _ in range(world_size)]
    dist.all_gather_object(all_shapes, tensor.shape)
    all_shapes = torch.tensor(all_shapes, device=tensor.device)
    return all_shapes


def all_gather_different_shapes(tensor):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Step 1 & 2: Gather tensor metadata
    local_metadata = {"shape": tensor.shape, "dtype": tensor.dtype}
    all_metadata = [None for _ in range(world_size)]
    dist.all_gather_object(all_metadata, local_metadata)

    # Step 3: Determine maximum size and pad local tensor
    max_shape = tuple(max(dim) for dim in zip(*[meta["shape"] for meta in all_metadata]))
    padded_tensor = pad_tensor(tensor, max_shape)

    # Step 4: All-gather padded tensors
    gathered_tensors = [
        torch.zeros(max_shape, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(gathered_tensors, padded_tensor)

    # Step 5: Unpad gathered tensors
    result = []
    for i, meta in enumerate(all_metadata):
        original_shape = meta["shape"]
        slices = tuple(slice(0, s) for s in original_shape)
        result.append(gathered_tensors[i][slices].clone())

    return result


def all_gather(tensor):
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors


def all_gather_list_tensor(local_list: List[torch.Tensor]) -> List[torch.Tensor]:
    world_size = dist.get_world_size()

    gathered_lists = [None] * world_size
    dist.all_gather(gathered_lists, local_list)

    flattened = [x for rank_list in gathered_lists for x in rank_list]

    return flattened


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left]
    )
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(
            left_rank, right_rank, tensor_to_left, tensor_to_right, group=group
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
        )


def neighbour_exchange_bidir_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    return NeighbourExchangeBidir.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


def pad_to_max(tensor, max_size):
    pad_size = max_size - tensor.size(1)
    if pad_size == 0:
        return tensor
    padding = torch.zeros(tensor.size(0), pad_size, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=1)


class DifferentiableAllGatherVarShapes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, world_size, rank):
        C, N_local = tensor.shape

        # Step 1: Gather local shapes
        all_N = [None] * world_size
        dist.all_gather_object(all_N, N_local)

        # Step 2: Flatten tensor
        flat = tensor.contiguous().view(-1)

        # Step 3: all_to_all
        input_list = [flat] * world_size
        output_list = [
            torch.empty(C * N_i, dtype=tensor.dtype, device=tensor.device) for N_i in all_N
        ]
        dist.all_to_all(output_list, input_list)

        # Step 4: Unflatten
        result = [chunk.view(C, N_i) for chunk, N_i in zip(output_list, all_N)]

        # Save for backward
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.C = C
        ctx.all_N = all_N
        ctx.input_shape = tensor.shape

        return result

    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_outputs: list of length world_size, each of shape C x N_i
        rank = ctx.rank
        C = ctx.C
        all_N = ctx.all_N

        # Step 1: Flatten grads
        grad_inputs_flat = [g.contiguous().view(-1) for g in grad_outputs]

        # Step 2: all_to_all (reverse direction)
        input_list = grad_inputs_flat
        output_tensor = torch.empty(
            C * all_N[rank], dtype=grad_outputs[0].dtype, device=grad_outputs[0].device
        )

        # Create dummy inputs on all ranks except self
        dummy_output_lists = [
            torch.empty(
                C * all_N[rank], dtype=grad_outputs[0].dtype, device=grad_outputs[0].device
            )
            for _ in range(ctx.world_size)
        ]
        dummy_output_lists[rank] = output_tensor

        dist.all_to_all(dummy_output_lists, input_list)

        # Reshape to original shape
        return output_tensor.view(ctx.input_shape), None, None


def differentiable_all_gather_varshapes(tensor, world_size, rank):
    return DifferentiableAllGatherVarShapes.apply(tensor, world_size, rank)
