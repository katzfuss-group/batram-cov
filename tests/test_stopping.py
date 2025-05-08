from functools import partial

from batram_cov.stopping import early_stopper


def test_stopping_increasing():
    patience, tol = 5, 1e-6
    stopper = partial(early_stopper, patience=patience, tol=tol)

    stop_state = (float("inf"), 0, None)

    for t in range(7):
        stop, stop_state = stopper(t, None, stop_state)
        if stop:
            break

    assert stop_state[1] == patience


def test_stopping():
    patience, tol = 2, 1e-6
    stopper = partial(early_stopper, patience=patience, tol=tol)

    stop_state = (float("inf"), 0, None)

    losses = [0, 1, 1, 1]
    for i, loss in enumerate(losses):
        stop, stop_state = stopper(loss, None, stop_state)
        if stop:
            break

    assert stop_state[1] == patience


def test_stopping_with_empty_init():
    patience, tol = 2, 1e-6
    stopper = partial(early_stopper, patience=patience, tol=tol)

    losses = [0, 1, 1, 1]
    _, stop_state = stopper(float("inf"), None)
    for i, loss in enumerate(losses):
        stop, stop_state = stopper(loss, None, stop_state)
        if stop:
            break

    assert stop_state[1] == patience


def test_long_stopping_run():
    patience, tol = 50, 1e-6
    stopper = partial(early_stopper, patience=patience, tol=tol)

    stop_state = (float("inf"), 0, None)

    for t in range(1000):
        stop, stop_state = stopper(t, None, stop_state)
        if stop:
            break

    assert stop_state[1] == patience


def test_stopping_converges():
    patience, tol = 50, 1e-6
    stopper = partial(early_stopper, patience=patience, tol=tol)

    stop_state = (float("inf"), 0, None)

    for t in range(1000):
        stop, stop_state = stopper(1 / (1 + t), None, stop_state)
        if stop:
            break

    assert stop_state[1] == 0
