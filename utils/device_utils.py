


"""
Device utility functions for the Advantech Intrusion Detection System.

This module provides utility functions for device-related operations including
user input handling with timeouts and object serialization checking.
"""

import base64
import pickle
import selectors
import sys
from typing import Any, Optional
import machineid



def get_input(prompt: str, timeout: int = 10) -> Optional[str]:
    """
    Prompt the user for input with a timeout.

    If the user doesn't provide input within the specified time, returns None.
    This function uses the selectors module to monitor standard input for a
    given period and returns the input if available.

    Args:
        prompt: The prompt message to display to the user.
        timeout: The maximum time (in seconds) to wait for input. Defaults to 10.

    Returns:
        The user's input as a string if provided within the timeout period,
        otherwise None if the timeout is reached.

    Example:
        >>> user_input = get_input("Enter your name: ", timeout=5)
        >>> if user_input:
        ...     print(f"Hello, {user_input}!")
        ... else:
        ...     print("No input received within timeout.")
    """
    # Display the prompt message to the user
    print(prompt)

    # Create a selector object to monitor standard input (stdin)
    selector = selectors.DefaultSelector()
    selector.register(sys.stdin, selectors.EVENT_READ)

    try:
        # Wait for input or timeout (timeout in seconds)
        events = selector.select(timeout)

        # If input is available before the timeout, return the input
        if events:
            return sys.stdin.readline().strip()

        # If no input is received within the timeout period, return None
        return None
    finally:
        # Clean up the selector
        selector.close()
    

def check_if_the_object_pickleable(obj: Any) -> bool:
    """
    Check if an object can be serialized using Python's pickle module.

    This function attempts to serialize the object using pickle.dumps().
    If the object is serializable, it returns True. If an exception occurs
    during serialization (such as if the object contains non-pickleable
    attributes), it returns False.

    Args:
        obj: The object to be checked for pickling capability.

    Returns:
        True if the object can be serialized (pickled), False otherwise.

    Example:
        >>> check_if_the_object_pickleable([1, 2, 3])
        True
        >>> check_if_the_object_pickleable(lambda x: x)
        False

    Note:
        This function is useful for checking multiprocessing compatibility,
        as objects passed between processes must be pickleable.
    """
    try:
        # Attempt to pickle the object
        pickle.dumps(obj)
        return True  # If no exception, the object is pickleable
    except Exception:  # pylint: disable=broad-except
        # If pickling fails, return False
        return False   

def check_the_system_and_get_uuid() -> Optional[str]:
    """
    Retrieves a unique hashed machine ID, encodes it in Base64, and returns it.

    Returns:
        Optional[str]: The Base64-encoded machine UUID if successful, otherwise None.
    """
    try:
        machine_id: str = machineid.id()
        
        
        # Convert the machine ID to bytes
        id_bytes: bytes = machine_id.encode()

        # Convert bytes to Base64 string
        id_base64: str = base64.urlsafe_b64encode(id_bytes).decode()
        
        return id_base64

    except Exception as e:
        print(f"Error retrieving machine UUID: {e}")
        return None