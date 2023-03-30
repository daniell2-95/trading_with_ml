from streamlit import session_state as _state

_PERSIST_STATE_KEY = f"{__name__}_PERSIST"

def persist(key: str) -> str:

    """
    Persists the given key in the Streamlit session state for future use.

    Parameters:
        key: The key to persist in the session state.

    Returns:
        The same key that was passed in, for convenience.
    """

    if _PERSIST_STATE_KEY not in _state:
        _state[_PERSIST_STATE_KEY] = set()

    _state[_PERSIST_STATE_KEY].add(key)

    return key


def load_widget_state() -> None:

    """
    Loads the persisted state for all widgets that have been persisted using the persist() function.
    If there is no persisted state, this function does nothing.

    Returns:
        None
    """

    if _PERSIST_STATE_KEY in _state:
        _state.update({
            key: value
            for key, value in _state.items()
            if key in _state[_PERSIST_STATE_KEY]
        })