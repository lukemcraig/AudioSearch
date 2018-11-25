from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter
import matplotlib.ticker as plticker


class PowerScale(mscale.ScaleBase):
    """
    """

    name = 'powerscale'

    def __init__(self, axis, power=.4, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.
        """
        mscale.ScaleBase.__init__(self)
        self.power = power

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.
        """
        return self.PowerScaleTransform(self.power)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.
        """

        class HzFormatter(Formatter):
            def __call__(self, x, pos=None):
                return "%dHz" % (x)

        axis.set_major_formatter(HzFormatter())
        axis.set_minor_formatter(HzFormatter())
        axis.set_major_locator(
            plticker.FixedLocator(
                [4, 10, 20, 30, 40, 60, 100, 200, 300, 400, 500, 600, 700, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
                 10000, 20000]
            ))
        axis.set_tick_params(rotation=-70)

    class PowerScaleTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, power):
            mtransforms.Transform.__init__(self)
            self.power = power

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy. Importantly, the
            ``transform`` method *must* return an array that is the
            same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """
            return a ** self.power

        def inverted(self):
            """
            Override this method so matplotlib knows how to get the
            inverse transform for this transform.
            """
            return PowerScale.InvertedPowerScaleTransform(self.power)

    class InvertedPowerScaleTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, power):
            mtransforms.Transform.__init__(self)
            # self.thresh = thresh
            self.power = power

        def transform_non_affine(self, a):
            return a ** (1.0 / self.power)

        def inverted(self):
            return PowerScale.PowerScaleTransform(self.power)
