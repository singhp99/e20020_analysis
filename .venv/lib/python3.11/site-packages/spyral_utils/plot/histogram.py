"""Modules for creating and plotting histograms in matplotlib

The default matplotlib histogramming tools can be hard to use for large datasets
as they require a lot of rebinning or all of the data to be loaded. Here we give some of our own
which allow for incremental filling of histograms

Classes
-------
Hist1D
    A 1-D histogram dataclass. Should not be instantiated directly
Hist2D
    A 2-D histogram dataclass. Should not be instantiated directly
Histogrammer
    A parent object used to create, manage histograms
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from math import floor


# Utility functions
def clamp_low(x: float, edge: float) -> float:
    return x if x > edge else edge


def clamp_hi(x: float, edge: float) -> float:
    return x if x < edge else edge


def clamp_range(xrange: tuple[float, float], min_max: tuple[float, float]):
    return (clamp_low(xrange[0], min_max[0]), clamp_hi(xrange[1], min_max[1]))


@dataclass
class Hist1D:
    """Dataclass wrapping a numpy array used to store histogram data and retrieve histogram statistics

    Attributes
    ----------
    name: str
        histogram name
    counts: ndarray
        array of histogram counts
    bins: ndarray
        array of histogram bin edges
    bin_width: float
        the width of histogram bins

    Methods
    -------
    get_bin(x: float) -> int | None
        get the bin number for an x-coordinate value
    stats_for_range(xrange: tuple[float, float]) -> tuple[float, float, float] | None
        get some statistics for a subrange of the histogram
    get_subrange(xrange: tuple[float, float]) -> tuple[ndarray, ndarray]
        get a histogram subrange (bin edges, counts)
    """

    name: str
    counts: NDArray[np.float64]
    bins: NDArray[np.float64]
    bin_width: float

    def get_bin(self, x: float) -> int | None:
        """Get the bin number which contains the x-coordinate

        Parameters
        ----------
        x: float
            X-coordinate for which we want to find the bin number

        Returns
        -------
        int | None
            The bin number or None if the x value does not fall within the histogram
        """
        if x < self.bins.min() or x > self.bins.max():
            return None

        return int(floor((x - self.bins[0]) / self.bin_width))

    def stats_for_range(
        self, xrange: tuple[float, float]
    ) -> tuple[float, float, float] | None:
        """Get some statistics for a histogram subrange

        Calculates the mean, integral, and standard deviation of the sub-range

        Parameters
        ----------
        xrange: tuple[float, float]
            the subrange of the histogram (min, max) in x-coordinates

        Returns
        -------
        tuple[float, float, float] | None
            Returns a tuple of (integral, mean, std. dev.) for the subrange, or None if the subrange is not within the histogram bounds

        """
        clamped_range = clamp_range(xrange, (self.bins.min(), self.bins.max()))
        bin_min = self.get_bin(clamped_range[0])
        bin_max = self.get_bin(clamped_range[1])
        if bin_min is None or bin_max is None:
            return None
        integral = np.sum(self.counts[bin_min:bin_max])
        mean = np.average(
            self.bins[bin_min:bin_max], weights=self.counts[bin_min:bin_max]
        )
        variance = np.average(
            (self.bins[bin_min:bin_max] - mean) ** 2.0,
            weights=self.counts[bin_min:bin_max],
        )
        return (integral, mean, np.sqrt(variance))  # type: ignore

    def get_subrange(
        self, xrange: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a subrange of the histogram

        Parameters
        ----------
        xrange: tuple[float, float]
            the subrange of the histogram (min, max) in x-coordinates

        Returns
        -------
        tuple[ndarray, ndarray]
            the subrange (bin edges, counts)
        """
        mask = np.logical_and(self.bins >= xrange[0], self.bins < xrange[1])
        return (self.bins[mask], self.counts[mask[:-1]])


@dataclass
class Hist2D:
    """Dataclass wrapping a numpy array used to store two-dimensional histogram data and retrieve histogram statistics

    Attributes
    ----------
    name: str
        histogram name
    counts: ndarray
        array of histogram counts
    x_bins: ndarray
        array of histogram x bin edges
    y_bins: ndarray
        array of histogram y bin edges
    x_bin_width: float
        the width of histogram x bins
    y_bin_width: float
        the width of histogram y bins

    Methods
    -------
    get_bin(coords: tuple[float, float]) -> tuple[int, int] | None
        get the x and y bin numbers for an (x,y)-coordinate value
    stats_for_range(xrange: tuple[float, float], yrange: tuple[float, float]) -> tuple[float, float, float, float, float] | None
        get some statistics for a subrange of the histogram
    get_subrange(xrange: tuple[float, float], xrange: tuple[float, float]) -> tuple[ndarray, ndarray, ndarray]
        get a histogram subrange (x bin edges, y bin edges, counts)
    """

    name: str
    counts: NDArray[np.float64]
    x_bins: NDArray[np.float64]
    y_bins: NDArray[np.float64]
    x_bin_width: float
    y_bin_width: float

    def get_bin(self, coords: tuple[float, float]) -> tuple[int, int] | None:
        """Get the x and y bin numbers for an (x,y)-coordinate value

        Parameters
        ----------
        coords: tuple[float, float]
            The (x,y) corrdinate for which we want to find the bin numbers

        Returns
        -------
        tuple[int, int] | None
            Returns the (x bin, y bin) numbers or None if out of range
        """
        if (coords[0] < self.x_bins.min() or coords[0] > self.x_bins.max()) or (
            coords[1] < self.y_bins.min() or coords[1] > self.y_bins.max()
        ):
            return None

        y_bin = int(floor((coords[1] - self.y_bins[0]) / self.y_bin_width))
        x_bin = int(floor((coords[0] - self.x_bins[0]) / self.x_bin_width))
        return (x_bin, y_bin)

    # returns (integral, mean x, std_dev x, mean y, std_dev y)
    def stats_for_range(
        self, xrange: tuple[float, float], yrange: tuple[float, float]
    ) -> tuple[float, float, float, float, float] | None:
        """Get some statistics for a histogram subrange

        Calculates the mean in x and y, integral, and standard deviation in x and y of the sub-range

        Parameters
        ----------
        xrange: tuple[float, float]
            the subrange of the histogram (min, max) in x-coordinates
        yrange: tuple[float, float]
            the subrange of the histogram (min, max) in y-coordinates

        Returns
        -------
        tuple[float, float, float, float, float] | None
            Returns a tuple of (integral, x mean, y mean, x std. dev., y std. dev.) for the subrange, or None if the subrange is not within the histogram bounds

        """
        clamped_x_range = clamp_range(xrange, (self.x_bins.min(), self.x_bins.max()))
        clamped_y_range = clamp_range(yrange, (self.y_bins.min(), self.y_bins.max()))
        bin_min = self.get_bin((clamped_x_range[0], clamped_y_range[0]))
        bin_max = self.get_bin((clamped_x_range[1], clamped_y_range[1]))
        if bin_min is None or bin_max is None:
            return None

        x_bin_range = np.arange(start=bin_min[0], stop=bin_max[0], step=1)
        y_bin_range = np.arange(start=bin_min[1], stop=bin_max[1], step=1)
        bin_mesh = np.ix_(y_bin_range, x_bin_range)

        integral = np.sum(self.counts[bin_mesh])
        mean_x = np.average(
            self.x_bins[bin_min[0] : bin_max[0]],
            weights=np.sum(self.counts.T[bin_min[0] : bin_max[0]], 1),
        )
        mean_y = np.average(
            self.y_bins[bin_min[1] : bin_max[1]],
            weights=np.sum(self.counts[bin_min[1] : bin_max[1]], 1),
        )
        var_x = np.average(
            (self.x_bins[bin_min[0] : bin_max[0]] - mean_x) ** 2.0,
            weights=np.sum(self.counts.T[bin_min[0] : bin_max[0]], 1),
        )
        var_y = np.average(
            (self.y_bins[bin_min[1] : bin_max[1]] - mean_y) ** 2.0,
            weights=np.sum(self.counts[bin_min[1] : bin_max[1]], 1),
        )
        return (integral, mean_x, mean_y, np.sqrt(var_x), np.sqrt(var_y))  # type: ignore

    def get_subrange(
        self, xrange: tuple[float, float], yrange: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a subrange of the histogram

        Parameters
        ----------
        xrange: tuple[float, float]
            the subrange of the histogram (min, max) in x-coordinates
        yrange: tuple[float, float]
            the subrange of the histogram (min, max) in y-coordinates

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            the subrange (x bin edges, y bin edges, counts)
        """
        x_mask = np.logical_and(self.x_bins >= xrange[0], self.x_bins < xrange[1])
        y_mask = np.logical_and(self.y_bins >= yrange[0], self.y_bins < yrange[1])
        bin_mesh = np.ix_(y_mask, x_mask)
        return (self.x_bins[x_mask], self.y_bins[y_mask], self.counts[bin_mesh])


class Histogrammer:
    """Histogrammer is a wrapper around a dictionary of str->Hist1D|Hist2D that interfaces with matplotlib

    A new histogram can be added to the dictionary using the add_hist1d/add_hist2d methods. The name passed to
    these methods is used as the key for the dictionary. To add data to the histograms use the fill_hist1d/fill_hist2d methods.
    The fill methods accept arrays of data, and this is by intent. It would not be efficient to fill the histograms point by point. Rather, prefer
    passing entire data sets (like dataframe columns). Finally, to retrieve a histogram (for plotting, etc), use the get_hist1d/get_hist2d methods.
    Prefer the getters over direct access to the underlying dictionary as the getters perfom some error checking.

    Attributes
    ----------
    histograms: dict[str, Hist1D | Hist2D]
        the histograms held by the Histogrammer, mapped by name

    Methods
    -------
    add_hist1d(name: str, bins: int, range: tuple[float, float])
        add a Hist1D
    add_hist2d(name: str, bins: tuple[int, int], ranges: tuple[tuple[float, float], tuple[float, float]])
        add a Hist2D
    fill_hist1d(self, name: str, data: ndarray) -> bool
        fill an existing Hist1D with some data
    fill_hist2d(self, name: str, x_data: ndarray, y_data: ndarray) -> bool
        fill an existing Hist2D with some data
    get_hist1d(name: str) -> Hist1D | None
        get a Hist1D by name
    get_hist2d(name: str) -> Hist2D | None
        get a Hist2D by name
    """

    def __init__(self):
        self.histograms: dict[str, Hist1D | Hist2D] = {}

    def add_hist1d(self, name: str, bins: int, range: tuple[float, float]):
        """Add a Hist1D to the Histogrammer

        Parameters
        ----------
        name: str
            The name of the histogram, it should be unqiue
        bins: int
            The number of bins
        range: tuple[float, float]
            The x-range of the histogram in x-axis coordinates
        """
        if name in self.histograms:
            print(f"Overwriting histogram named {name} in Histogrammer.add_histogram!")

        hist = Hist1D(
            name, np.empty(0), np.empty(0), np.abs(range[0] - range[1]) / float(bins)
        )
        hist.counts, hist.bins = np.histogram(a=[], bins=bins, range=range)
        self.histograms[name] = hist

    def add_hist2d(
        self,
        name: str,
        bins: tuple[int, int],
        ranges: tuple[tuple[float, float], tuple[float, float]],
    ):
        """Add a Hist2D to the Histogrammer

        Parameters
        ----------
        name: str
            The name of the histogram, it should be unqiue
        bins: tuple[int, int]
            The number of (x bins, y bins)
        ranges: tuple[tuple[float, float], tuple[float, float]]
            The range of the histogram ((min x, max x), (min y, max y))
        """
        if name in self.histograms:
            print(f"Overwriting histogram named {name} in Histogrammer.add_histogram!")

        hist = Hist2D(
            name,
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.abs(ranges[0][0] - ranges[0][1]) / float(bins[0]),
            np.abs(ranges[1][0] - ranges[1][1]) / float(bins[1]),
        )
        hist.counts, hist.x_bins, hist.y_bins = np.histogram2d(
            x=[], y=[], bins=bins, range=ranges
        )
        hist.counts = hist.counts.T
        self.histograms[name] = hist

    def fill_hist1d(self, name: str, data: np.ndarray) -> bool:
        """Fill a Hist1D with some data

        Parameters
        ----------
        name: str
            The name of the Hist1D
        data: ndarray
            The data to fill the histogram with. Should be a 1-D array

        Returns
        -------
        bool
            Indicates if data was successfully added to the histogram

        """
        if name not in self.histograms:
            return False

        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return False

        hist.counts = hist.counts + np.histogram(a=data, bins=hist.bins)[0]
        return True

    def fill_hist2d(self, name: str, x_data: np.ndarray, y_data: np.ndarray) -> bool:
        """Fill a Hist1D with some data

        The parameters x_data and y_data should have the same length.

        Parameters
        ----------
        name: str
            The name of the Hist1D
        x_data: ndarray
            The x coordinates of the data. Should be a 1-D array.
        y_data: ndarray
            The y coordinates of the data. Should be a 1-D array.

        Returns
        -------
        bool
            Indicates if data was successfully added to the histogram

        """
        if name not in self.histograms:
            return False

        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return False
        counts, _, _ = np.histogram2d(
            x_data.flatten(),
            y_data.flatten(),
            bins=(hist.x_bins, hist.y_bins),  # type: ignore
        )
        hist.counts += counts.T
        return True

    def get_hist1d(self, name: str) -> Hist1D | None:
        """Retrieve a Hist1D by name

        Parameters
        ----------
        name: str
            The name of the histogram

        Returns
        -------
        Hist1D | None
            Returns Hist1D if a Hist1D exists with the given name. Otherwise returns None.

        """
        if name not in self.histograms:
            return None

        hist = self.histograms[name]
        if type(hist) is not Hist1D:
            return None
        else:
            return hist

    def get_hist2d(self, name: str) -> Hist2D | None:
        """Retrieve a Hist2D by name

        Parameters
        ----------
        name: str
            The name of the histogram

        Returns
        -------
        Hist2D | None
            Returns Hist2D if a Hist2D exists with the given name. Otherwise returns None.

        """
        if name not in self.histograms:
            return None

        hist = self.histograms[name]
        if type(hist) is not Hist2D:
            return None
        else:
            return hist
