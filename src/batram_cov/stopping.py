from copy import deepcopy
from typing import Any

type Params = Any
StopState = tuple[float, int, Params]


def early_stopper(
    val: float,
    params: Params,
    stop_state: StopState | None = None,
    *,
    warmup_phase: int = 0,
    patience: int = 0,
    tol: float = 0.0,
) -> tuple[bool, StopState]:
    """Implements a simple early stopping mechanism.

    Args:
    -----
    :param:`val (float)`
        The current loss to use in comparison.

    :param:`stop_state (tuple[float, int])`
        A tuple (best_loss, counter) to keep track of the best loss and number
        of epochs to wait before stopping training. If the counter passes the
        patience threshold, a stop flag is returned as True.

    :param:`patience (int)`
        The number of epochs to wait before stopping training.

    Returns:
    --------
    :param:`stop (bool)`
        A signal to break the training loop.

    :param:`stop_state (tuple[float, int])`
        The updated stop_state tuple.
    """


    if stop_state is None:
        stop_state = (float('inf'), 0, 0, 0, params)
        stop = 0 < patience
        return stop, stop_state

    best_val, abs_counter, counter, resets, best_params = stop_state
    abs_counter = abs_counter + 1

    if best_params is None:
        best_params = deepcopy(params)

    if abs_counter <= warmup_phase:
        val = float('inf')
        counter = 0

    if val < best_val + tol:
        # if we see a value within the tolerance, we reset the counter
        # if we would allow the best_val to be updated,
        # we would allow the best_val to creap up
        counter = 0
        resets += 1

    if val < best_val:
        best_val = val
        best_params = deepcopy(params)

    else:
        counter += 1
    
    stop_state = (best_val, abs_counter, counter, resets, best_params)
    stop = counter >= patience

    return stop, stop_state
