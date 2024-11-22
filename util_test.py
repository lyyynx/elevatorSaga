from util import list_to_string


def test_list_to_string() -> None:
    assert list_to_string([]) == ""
    assert list_to_string([[1, 2], [3, 4]]) == "[[1,2],[3,4]]"
    assert list_to_string([-1.0, 1.0]) == "[-1.0,1.0]"
