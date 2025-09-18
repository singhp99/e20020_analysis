"""Module conatining methods for creating, saving, and using graphical Cuts on data

Classes
-------
CutHandler
    Handler to recieve vertices from a matplotlib or plotly selection event.
Cut2D
    Implementation of 2D cuts/gates/selections as used in many types of graphical analyses

Functions
---------
serialize_cut(cut: Cut2D, filepath: Path) -> bool
    Serialize cut to JSON and write to a file
deserialize_cut(filepath: Path) -> Cut2D | None
    Deserialize the JSON representation of a Cut2D from a file
"""

from polars import Series
from shapely import Polygon, Point, contains_xy
import numpy as np
import json
from pathlib import Path
from typing import Any

DEFAULT_CUT_AXIS = "DefaultAxis"


class CutHandler:
    """Handler to recieve vertices from a matplotlib or plotly selector.

    Typically will be used interactively. The appropriate on_select method should be passed to the selector object or callback for the
    plotting API used. CutHandler currently supports matplotlib and plotly. CutHandler can also be used in analysis applications to store cuts.

    An example script for each API:

    Matplotlib
    ```python
    from spyral_utils.plot import CutHandler, Cut2D, serialize_cut
    from matplotlib.widgets import PolygonSelector
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)
    handler = CutHandler()
    selector = PolygonSelector(ax, handler.mpl_on_select)

    # Plot some data here...

    plt.show()

    # Wait for user to draw a cut and close the window

    my_cut = handler.cuts["cut_0"]
    my_cut.name = "my_cut"
    serialize_cut(mycut, "my_cut.json")
    ```

    Plotly
    ```python
    from spyral_utils.plot import CutHandler, Cut2D, serialize_cut
    import plotly.graph_objects as go

    handler = CutHandler()

    # Do some plotting

    fig = go.Figure()
    fig.add_trace(...)

    # Bind the callback

    my_plot = fig.data[0]
    my_plot.on_select(handler.plotly_on_select)

    # Wait for user selection

    my_cut = handler.cuts["cut_0"]
    my_cut.name = "my_cut"
    serialize_cut(my_cut, "my_cut.json")

    ```

    Attributes
    ----------
    cuts: dict[str, Cut2D]
        mapping of cut name to Cut2D

    Methods
    -------
    mpl_on_select(verticies: list[tuple[float, float]])
        recieve a matplotlib polygon and create a Cut2D from it
    plotly_on_select(trace: Any, points: Any, selector: Any)
        recieve a plotly selection event and create a Cut2D from it
    """

    def __init__(self):
        self.cuts: dict[str, Cut2D] = {}

    def mpl_on_select(self, vertices: list[tuple[float, float]]):
        """Callback for use with matplotlib

        Parameters
        ----------
        vertices: list[tuple[float, float]]
            polygon vertices
        """
        cut_default_name = f"cut_{len(self.cuts)}"
        self.cuts[cut_default_name] = Cut2D(cut_default_name, vertices)

    def plotly_on_select(self, trace: Any, points: Any, selector: Any):
        """Callback for use with plotly

        Parameters
        ----------
        trace: Any
            The plotly trace from which the event originated (not relevant)
        points:
            A plotly Points object containing the data indicies within the selection (not relevant)
        selector:
            The selector object (either BoxSelector or LassoSelector)
        """
        if len(selector.xs) < 2:
            return

        cut_default_name = f"cut_{len(self.cuts)}"
        self.cuts[cut_default_name] = Cut2D(
            cut_default_name, list(zip(selector.xs, selector.ys))
        )


class Cut2D:
    """Implementation of 2D cuts/gates/selections as used in many types of graphical analyses

    Uses Shapely Polygon objects. Takes in a name (to identify the cut) and a list of points. The Polygon
    takes the verticies, and can then be used to check if a point(s) is inside of the polygon using the
    contains_* functions. Can be serialized to json format. Can also retreive Nx2 ndarray of vertices
    for plotting after the fact.

    Attributes
    ----------
    path: matplotlib.path.Path
        A matplotlib path (polygon) that is the actual cut shape
    name: str
        A name for the cut
    x_axis: str
        A name for the x-coordinate axis. Can be used with dataframes
        to programmatically specify what data should be used to process
        a cut.
    y_axis: str
        A name for the y-coordinate axis. Can be used with dataframes
        to programmatically specify what data should be used to process
        a cut.

    Methods
    -------
    is_point_inside(x: float, y: float) -> bool
        Check if a single point (x,y) is inside the cut
    is_arr_inside(points: list[tuple[float, float]]) -> list[bool]
        Check if a list of points (x,y) are inside the cut
    is_cols_inside(columns: Series) -> Series
        Check if a set of polars Columns are inside the cut
    get_vertices() -> ndarray
        Get the cut vertices
    serialize_json() -> str
        Get the JSON representation of the cut
    """

    def __init__(
        self,
        name: str,
        vertices: list[tuple[float, float]],
        x_axis: str = DEFAULT_CUT_AXIS,
        y_axis: str = DEFAULT_CUT_AXIS,
    ):
        self.polygon: Polygon = Polygon(vertices)
        self.name = name
        self.x_axis = x_axis
        self.y_axis = y_axis

    def is_point_inside(self, x: float, y: float) -> bool:
        """Is a point in the cut

        Parameters
        ----------
        x: float
            point x-coordinate
        y: float
            point y-coordinate

        Returns
        -------
        bool
            true if inside, false if outside
        """
        return self.polygon.contains(Point(x, y))

    def is_arr_inside(self, points: list[tuple[float, float]]) -> list[bool]:
        """Which of the points in this list are in the cut

        Parameters
        ----------
        points: list[tuple[float, float]]
            List of points (x,y)

        Returns
        -------
        list[bool]
            List of results of checking each point
        """
        return [contains_xy(self.polygon, point) for point in points]  # type: ignore

    def is_cols_inside(self, columns: Series) -> Series:
        """Which of the points in this Series are in the cut

        Parameters
        ----------
        columns: Series
            Polars dataframe series to check

        Returns
        -------
        Series
            Series of True or False for each point
        """
        data = np.transpose(
            [columns.struct.field(name).to_list() for name in columns.struct.fields]
        )
        return Series(
            [bool(contains_xy(self.polygon, x=point[0], y=point[1])) for point in data]
        )

    def get_vertices(self) -> list[tuple[float, float]]:
        """Get the cut vertices

        Returns
        -------
        list[tuple]
            the vertices

        """
        return tuple(self.polygon.exterior.coords)  # type: ignore

    def get_x_axis(self) -> str:
        """Get the name of the cut data x-axis

        The x-axis is the name of the data used to form
        the cut along the x-axis. Typically, this is the name
        of a dataframe column.

        Returns
        -------
        str
            The name of the x-axis. By default, the value
            is DefaultAxis.
        """
        return self.x_axis

    def get_y_axis(self) -> str:
        """Get the name of the cut data y-axis

        The y-axis is the name of the data used to form
        the cut along the y-axis. Typically, this is the name
        of a dataframe column.

        Returns
        -------
        str
            The name of the y-axis. By default, the value
            is DefaultAxis.
        """
        return self.y_axis

    def is_default_x_axis(self) -> bool:
        """Check if the x-axis is the default value

        If the x-axis field is unset, it defaults to DefaultAxis.

        Returns
        -------
        bool
            True if default, False otherwise
        """
        return self.x_axis == DEFAULT_CUT_AXIS

    def is_default_y_axis(self) -> bool:
        """Check if the y-axis is the default value

        If the y-axis field is unset, it defaults to DefaultAxis.

        Returns
        -------
        bool
            True if default, False otherwise
        """
        return self.y_axis == DEFAULT_CUT_AXIS

    def serialize_json(self) -> str:
        """Serialize to JSON

        Returns
        -------
        str
            JSON representation
        """
        return json.dumps(
            self,
            default=lambda obj: {
                "name": obj.name,
                "xaxis": obj.x_axis,
                "yaxis": obj.y_axis,
                "vertices": tuple(obj.polygon.exterior.coords),
            },
            indent=4,
        )


def serialize_cut(cut: Cut2D, filepath: Path) -> bool:
    """Serialize the cut to JSON and write to a file file

    Parameters
    ----------
    cut: Cut2D
        Cut to serialize
    filepath: Path
        Path at which cut should be written

    Returns
    -------
    bool
        True on success, False on failure
    """
    json_str = cut.serialize_json()
    try:
        with open(filepath, "w") as output:
            output.write(json_str)
            return True
    except OSError as error:
        print(f"An error occurred writing cut {cut.name} to file {filepath}: {error}")
        return False


def deserialize_cut(filepath: Path) -> Cut2D | None:
    """Deserialize the JSON representation of a Cut2D

    Parameters
    ----------
    filepath: Path
        Path at which cut should be read from

    Returns
    -------
    Cut2D | None
        Returns a Cut2D on success, None on failure
    """
    try:
        with open(filepath, "r") as input:
            buffer = input.read()
            cut_dict = json.loads(buffer)
            if not ("name" in cut_dict and "vertices" in cut_dict):
                print(
                    f"Data in file {filepath} is not the right format for Cut2D, could not load"
                )
                return None
            xaxis = DEFAULT_CUT_AXIS
            yaxis = DEFAULT_CUT_AXIS
            if "xaxis" in cut_dict and "yaxis" in cut_dict:
                xaxis = cut_dict["xaxis"]
                yaxis = cut_dict["yaxis"]
            return Cut2D(
                cut_dict["name"],
                cut_dict["vertices"],
                x_axis=xaxis,
                y_axis=yaxis,
            )
    except Exception as error:
        print(
            f"An error occurred reading trying to read a cut from file {filepath}: {error}"
        )
        return None
