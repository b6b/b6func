"""
b6func.py

Various VapourSynth functions I find useful

Dependencies for full functionality:
- VapourSynth r39+:   https://github.com/vapoursynth/vapoursynth
- descale:            https://github.com/Irrational-Encoding-Wizardry/vapoursynth-descale
- eedi3(cl):          https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI3
- f3kdb:              https://forum.doom9.org/showthread.php?t=161411
- fmtconv:            https://forum.doom9.org/showthread.php?t=166504
- fvsfunc.py:         https://github.com/Irrational-Encoding-Wizardry/fvsfunc
- havsfunc.py:        https://github.com/HomeOfVapourSynthEvolution/havsfunc
- kagefunc.py:        https://github.com/Irrational-Encoding-Wizardry/kagefunc
- nnedi3cl:           https://github.com/HomeOfVapourSynthEvolution/VapourSynth-NNEDI3CL
- znedi3:             https://github.com/sekrit-twc/znedi3

"""

from functools import partial
from math import ceil, log

import vapoursynth as vs

import fvsfunc as fvf
import havsfunc as hvf
import kagefunc as kgf

core = vs.core


def mf3kdb(src, mask=None, range=15, y=40, cb=None, cr=None, agrain=0, luma_scaling=12,
           grainy=0, grainc=0, blur_first=True, keep_tv_range=False, output_depth=None):
    """
    Masked f3kdb

    A wrapper function for f3kdb that can optionally merge with an external mask clip
    and additionally add kagefunc's adaptive_grain to the final debanded clip.

    grainy, grainc, and adaptive_grain are applied to the final, post-merged clip.

    Some f3kdb default behavior is also different:
    - y defaults to 40 and cr/cb default to y//2
    - output_depth defaults to the source bit depth
    - grainy and grainc default to 0
    - grain is always static

    Parameters:
    -----------
    Most f3kdb arguments (https://f3kdb.readthedocs.io/en/latest/index.html)
    mask:                mask clip to use for merging with debanded clip
    agrain (0):          strength value for adaptive_grain
    luma_scaling (12):   luma_scaling value for adaptive_grain

    """
    func = 'mf3kdb'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(func + ": 'src' must be a clip")
    if mask is not None and not isinstance(mask, vs.VideoNode):
        raise TypeError(func + ": 'mask' must be a clip")

    src_bits = src.format.bits_per_sample

    if cb is None:
        cb = y // 2
    if cr is None:
        cr = y // 2
    if output_depth is None:
        output_depth = src_bits
    if mask is not None and mask.format.bits_per_sample != src_bits:
        mask = fvf.Depth(mask, src_bits, dither_type='none')

    out = core.f3kdb.Deband(src, range=range, y=y, cb=cb, cr=cr, grainy=grainy, grainc=grainc,
                            blur_first=blur_first, keep_tv_range=keep_tv_range, output_depth=src_bits)

    if mask is not None:
        out = core.std.MaskedMerge(out, src, mask)
    if grainy > 0 or grainc > 0:
        out = core.f3kdb.Deband(out, range=0, y=0, cb=0, cr=0, grainy=grainy, grainc=grainc,
                                keep_tv_range=keep_tv_range, output_depth=src_bits)
    if agrain > 0:
        out = kgf.adaptive_grain(out, strength=agrain, luma_scaling=luma_scaling)

    return fvf.Depth(out, output_depth)


def rescale(src, w, h, mask_detail=False, mask=None, aatype=None, thr=10, expand=2, inflate=2,
            descale_kernel='bicubic', b=1/3, c=1/3, descale_taps=3, kernel='spline36',
            taps=None, invks=False, invkstaps=3, a1=None, a2=None, nsize=4, nns=4,
            ocl_aa=None, ocl_upscale=False, f=None, show_mask=False):
    """
    Descale and re-upscale a clip

    This descales a clip's luma, nnedi3 resamples it back to its original resolution,
    and merges back in the original chroma if applicable. It can also mask detail and merge
    it back into the final rescaled clip (and optionally anti-alias that detail too).
    It will opt to use opencl for situations where it will be faster (mainly eedi3 aa).
    There are separate parameters for opencl aa and upscaling because if you use both together
    you may run out of memory on lower-end gpus while doing heavy gpu filtering.

    Parameters:
    -----------
    w:                           source clip's native width to descale to
    h:                           source clip's native height to descale to
    mask_detail (False):         mask higher-than-native-resolution detail
    mask:                        mask clip to use instead of built-in mask generation
    aatype:                      aa type to use on masked detail ('nnedi3', 'eedi3')
    thr (10):                    threshold for built-in mask generation
    expand (2):                  number of times to expand built-in mask
    inflate (2):                 number of times to inflate built-in mask
    descale_kernel ('bicubic'):  kernel for descale
    b (1/3):                     b value for descale
    c (1/3):                     c value for descale
    descale_taps (3):            taps value for descale
    kernel ('spline36'):         kernel for nnedi3 resample rescale
    taps:                        taps value for nnedi3 resample rescale
    invks (False):               invks for nnedi3 resample rescale
    invkstaps (3):               invkstaps for nnedi3 resample
    a1:                          a1 for nnedi3 resample
    a2:                          a2 for nnedi3 resample
    nsize (4):                   nsize for nnedi3 resample
    nns (4):                     nns for nnedi3 resample
    ocl_aa:                      use opencl vatiants of nnedi3/eedi3 for anti-aliasing
    ocl_upscale (False):         use opencl variant of nnedi3 for luma upscale
    f:                           function to perform on descaled luma while in native resolution
    show_mask (False):           output detail mask

    """
    func = 'rescale'

    def conditional_aa(n, f, y, rescaled):
        diff = f.props.PlaneStatsDiff

        if diff > 0:
            return simple_aa(y, aatype=aatype, ocl=ocl_aa)

        return rescaled

    valid_aatypes = ['nnedi3', 'eedi3']

    if isinstance(aatype, str):
        aatype = aatype.lower()
    if not isinstance(src, vs.VideoNode):
        raise TypeError(func + ": 'src' must be a clip")
    if mask is not None and not isinstance(mask, vs.VideoNode):
        raise TypeError(func + ": 'mask' must be a clip")
    if w is None or h is None:
        raise TypeError(func + ": native 'w' and 'h' must be given")
    if show_mask and not mask_detail:
        raise TypeError(func + ": 'show_mask' can only be used with mask_detail=True")
    if aatype is not None and not mask_detail:
        raise TypeError(func + ": 'aatype' can only be set with mask_detail=True")
    if aatype is not None and aatype not in valid_aatypes:
        raise TypeError(func + ": 'aatype' must be 'nnedi3' or 'eedi3'")

    sw = src.width
    sh = src.height
    src_bits = src.format.bits_per_sample
    is_gray = src.format.color_family == vs.GRAY

    if aatype is not None and ocl_aa is None:
        ocl_aa = True if aatype == 'eedi3' else False
    if mask is not None and mask.format.bits_per_sample != src_bits:
        mask = fvf.Depth(mask, src_bits, dither_type='none')

    nnrs = nnedi3cl_resample if ocl_upscale else nnedi3_resample

    y = src if is_gray else getY(src)
    descaled = fvf.Resize(y, w, h, kernel=descale_kernel, a1=b, a2=c, taps=descale_taps, invks=True)

    # Built-in diff mask generation (from fvsfunc's DescaleM)
    if mask_detail and mask is None:
        peak = (1 << src_bits) - 1
        thr = hvf.scale(thr, peak)

        up = fvf.Resize(descaled, sw, sh, kernel=descale_kernel, a1=b, a2=c, taps=descale_taps)

        diff_mask = core.std.Expr([y, up], 'x y - abs')
        diff_mask = fvf.Resize(diff_mask, w, h, kernel='bilinear')
        diff_mask = core.std.Binarize(diff_mask, threshold=thr)

        diff_mask = kgf.iterate(diff_mask, core.std.Maximum, expand)
        diff_mask = kgf.iterate(diff_mask, core.std.Inflate, inflate)

        diff_mask = core.resize.Spline36(diff_mask, sw, sh)
    elif mask_detail:
        diff_mask = mask

    if show_mask:
        return diff_mask

    if f is not None:
        descaled = f(descaled)

    rescaled = nnrs(descaled, sw, sh, nsize=nsize, nns=nns, kernel=kernel,
                    a1=a1, a2=a2, taps=taps, invks=invks, invkstaps=invkstaps)

    if aatype is not None:
        blank = core.std.BlankClip(diff_mask)
        mask_stats = core.std.PlaneStats(diff_mask, blank)
        nonscaled = core.std.FrameEval(rescaled, partial(conditional_aa, y=y, rescaled=rescaled),
                                       prop_src=mask_stats)
    else:
        nonscaled = y

    if mask_detail:
        rescaled = core.std.MaskedMerge(rescaled, nonscaled, diff_mask)

    if is_gray:
        return rescaled

    return merge_chroma(rescaled, src)


# Masked rescale aliases
masked_rescale = partial(rescale, mask_detail=True)     # PEP 8
rescaleM = partial(rescale, mask_detail=True)           # AVS-like name style


def edi_resample(src, w, h, edi=None, kernel='spline36', a1=None, a2=None,
                 invks=False, taps=4, invkstaps=4, **kwargs):
    """
    Edge-directed interpolation resampler

    Doubles the height with the given edge-directed interpolation filter as many times
    as needed and downsamples to the given w and h, fixing the chroma shift if necessary.
    
    Supports:
    - eedi2
    - eedi3
    - eedi3cl
    - nnedi3 (znedi3)
    - nnedi3cl

    Currently, it only correctly supports maintaining a similar aspect ratio
    because it always doubles both the width and height.
    """
    func = 'edi_resample'

    valid_edis = {
        'eedi2': ['mthresh', 'lthresh', 'vthresh', 'estr', 'dstr', 'maxd', 'map', 'nt', 'pp'],
        'eedi3': ['alpha', 'beta', 'gamma', 'nrad', 'mdis', 'hp', 'ucubic', 'cost3', 'vcheck',
                  'vthresh0', 'vthresh1', 'vthresh2', 'sclip', 'opt'],
        'eedi3cl': ['alpha', 'beta', 'gamma', 'nrad', 'mdis', 'hp', 'ucubic', 'cost3', 'vcheck',
                    'vthresh0', 'vthresh1', 'vthresh2', 'sclip', 'opt', 'device'],
        'nnedi3': ['nsize', 'nns', 'qual', 'etype', 'pscrn', 'opt', 'int16_prescreener',
                   'int16_predictor', 'exp'],
        'nnedi3cl': ['nsize', 'nns', 'qual', 'etype', 'pscrn', 'device'],
    }

    if not isinstance(src, vs.VideoNode):
        raise TypeError(func + ": 'src' must be a clip")
    if not isinstance(edi, str):
        raise TypeError(func + ": Must use a supported edge-directed interpolation filter string")

    edi = edi.lower()
    
    if edi not in valid_edis:
        raise TypeError(func + ": '" + edi + "' is not a supported edge-directed interpolation filter")

    # Check if kwargs are valid for given edi
    for arg in kwargs:
        if arg not in valid_edis[edi]:
            raise TypeError(func + ": '" + arg + "' is not a valid argument for " + edi)

    edifunc = {
        'eedi2': (lambda src: core.eedi2.EEDI2(src, field=1, **kwargs).std.Transpose()),
        'eedi3': (lambda src: core.eedi3m.EEDI3(src, field=1, dh=True, **kwargs).std.Transpose()),
        'eedi3cl': (lambda src: core.eedi3m.EEDI3CL(src, field=1, dh=True, **kwargs).std.Transpose()),
        'nnedi3': (lambda src: core.znedi3.nnedi3(src, field=1, dh=True, **kwargs).std.Transpose()),
        'nnedi3cl': (lambda src: core.nnedi3cl.NNEDI3CL(src, field=1, dh=True, dw=True, **kwargs)),
    }

    scale = h / src.height

    if scale == 1:
        return src

    double_count = ceil(log(scale, 2))
    double_count = double_count * 2 if edi != 'nnedi3cl' else double_count

    doubled = src

    for _ in range(double_count):
        doubled = edifunc[edi](doubled)

    sx = [-0.5, -0.5 * src.format.subsampling_w] if double_count >= 1 else 0
    sy = [-0.5, -0.5 * src.format.subsampling_h] if double_count >= 1 else 0

    down = core.fmtc.resample(doubled, w=w, h=h, sx=sx, sy=sy, kernel=kernel,
                              a1=a1, a2=a2, taps=taps, invks=invks, invkstaps=invkstaps)

    return fvf.Depth(down, src.format.bits_per_sample)


# Aliases for various edge-directed interpolation filters to use with edi_resample
eedi2_resample = partial(edi_resample, edi='eedi2', mthresh=None, lthresh=None, vthresh=None,
                         estr=None, dstr=None, maxd=None, map=None, nt=None, pp=None)

eedi3_resample = partial(edi_resample, edi='eedi3', alpha=None, beta=None, gamma=None,
                         nrad=None, mdis=None, hp=None, ucubic=None, cost3=None, vcheck=None,
                         vthresh0=None, vthresh1=None, vthresh2=None, sclip=None)

eedi3cl_resample = partial(edi_resample, edi='eedi3cl', alpha=None, beta=None, gamma=None,
                           nrad=None, mdis=None, hp=None, ucubic=None, cost3=None, vcheck=None,
                           vthresh0=None, vthresh1=None, vthresh2=None, sclip=None)

nnedi3_resample = partial(edi_resample, edi='nnedi3', nsize=4, nns=4, qual=2, etype=None, pscrn=None,
                          opt=None, int16_prescreener=None, int16_predictor=None, exp=None)

nnedi3cl_resample = partial(edi_resample, edi='nnedi3cl', nsize=4, nns=4, qual=2,
                            etype=None, pscrn=None, device=None)


def simple_aa(src, aatype='nnedi3', mask=None, ocl=None, nsize=3, nns=1,
              qual=2, alpha=0.5, beta=0.2, nrad=3, mdis=30):
    """
    Basic nnedi3/eedi3 anti-aliasing

    Additionally includes optional masking using an external mask.
    Default values should be good for most anti-aliasing.
    Currently only anti-aliases YUV luma/GRAY.

    Parameters:
    -----------
    aatype ('nnedi3'): type of anti-aliasing to use ('nnedi3', 'eedi3')
    mask:              optional, external mask to use
    ocl:               use opencl variants of nnedi3/eedi3
    relevant nnedi3/eedi3 specific parameters

    """
    func = 'simple_aa'

    valid_aatypes = ['nnedi3', 'eedi3']

    if isinstance(aatype, str):
        aatype = aatype.lower()
    if aatype not in valid_aatypes:
        raise TypeError(func + ": 'aatype' must be 'nnedi3' or 'eedi3'")

    sw = src.width
    sh = src.height
    src_bits = src.format.bits_per_sample
    is_gray = src.format.color_family == vs.GRAY

    if ocl is None:
        ocl = True if aatype == 'eedi3' else False

    nnedi3 = core.nnedi3cl.NNEDI3CL if ocl else core.znedi3.nnedi3
    eedi3 = core.eedi3m.EEDI3CL if ocl else core.eedi3m.EEDI3

    if mask is not None and mask.format.bits_per_sample != src_bits:
        mask = fvf.Depth(mask, src_bits, dither_type='none')

    y = src if is_gray else getY(src)

    if aatype == 'nnedi3':
        if ocl:
            aa = nnedi3(y, field=1, dh=True, dw=True, nsize=nsize, nns=nns, qual=qual)
        else:
            aa = nnedi3(y, field=1, dh=True, nsize=nsize, nns=nns, qual=qual).std.Transpose()
            aa = nnedi3(aa, field=1, dh=True, nsize=nsize, nns=nns, qual=qual).std.Transpose()

        aa = core.fmtc.resample(aa, w=sw, h=sh, sx=-0.5, sy=-0.5)
    elif aatype == 'eedi3':
        aa = eedi3(y, field=1, dh=True, alpha=alpha, beta=beta, nrad=nrad, mdis=mdis).std.Transpose()
        aa = eedi3(aa, field=1, dh=True, alpha=alpha, beta=beta, nrad=nrad, mdis=mdis).std.Transpose()
        aa = core.fmtc.resample(aa, w=sw, h=sh, sx=-0.5, sy=-0.5)

    aa = fvf.Depth(aa, src_bits)    # fmtc.resample y u no always output same depth

    if mask is not None:
        aa = core.std.MaskedMerge(y, aa, mask)

    if is_gray:
        return aa

    return merge_chroma(aa, src)


def diff_mask(src, ref, thr=10, expand=4, inflate=4, blur=2):
    """
    Mask containing the difference between two clips according to thr

    A general purpose mask that checks the difference in all planes between the given
    src and ref clips. It can be useful for masking credits from NC content and also
    any colored hardsubbed content (regardless of edges like harsubmask_fades).

    I found the default values useful for masking complex/intricate OP/ED credits,
    but for simple hardsubbed stuff you can push thr higher to have less false positives.

    Parameters:
    -----------
    src:         clip with credits/hardsubs/whatever you want to mask
    ref:         reference clip
    thr (10):    threshold for detecting differences (lower = more included in mask)
    expand (4):  number of times to expand the mask
    inflate (4): number of times to inflate the mask
    blur (2):    number of times to blur the clips before comparing (to minimize false positives)

    """
    peak = (1 << src.format.bits_per_sample) - 1
    thr = hvf.scale(thr, peak)

    src = core.resize.Point(src, format=vs.YUV444P16)
    ref = core.resize.Point(ref, format=vs.YUV444P16)

    src = core.std.BoxBlur(src, hradius=1, hpasses=blur, vradius=1, vpasses=blur)
    ref = core.std.BoxBlur(ref, hradius=1, hpasses=blur, vradius=1, vpasses=blur)

    mask = core.std.Expr([src, ref], 'x y - abs')
    mask = core.std.Binarize(mask, threshold=thr)
    mask = kgf.iterate(mask, core.std.Maximum, expand)
    mask = kgf.iterate(mask, core.std.Inflate, inflate)

    mask_planes = getYUV(mask)

    out = core.std.Expr(mask_planes, 'x y + z +')

    return out


def border_mask(src, left=0, right=0, top=0, bottom=0):
    """
    Create a mask with the same clip properties as the given source clip and the
    number of pixels included from each edge. Useful for fixing edges when interpolation
    doesn't cut it or when you need to do other random things to edges.

    Parameters:
    -----------
    left (0):   number of pixels from the edge to include in mask
    right (0):  "
    top (0):    "
    bottom (0): "

    """
    bits = src.format.bits_per_sample
    white = 1 if src.format.sample_type == vs.FLOAT else (1 << bits) - 1
    mask_format = src.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)

    out = core.std.BlankClip(src, _format=mask_format)
    out = core.std.Crop(out, left=left, right=right, top=top, bottom=bottom)
    out = core.std.AddBorders(out, left=left, right=right, top=top, bottom=bottom, color=[white])

    return out


def select_range_every(src, cycle=1500, length=50, offset=0):
    """ VapourSynth SelectRangeEvery equivalent """
    return core.std.SelectEvery(src, cycle=cycle, offsets=range(offset, offset + length)).std.AssumeFPS(src)


sre = select_range_every    # alias


def merge_chroma(src, ref):
    """ Merge chroma from ref onto src """
    return core.std.ShufflePlanes([src, ref], planes=[0, 1, 2], colorfamily=ref.format.color_family)


def getY(src):
    """ Get Y plane from YUV clip """
    return core.std.ShufflePlanes(src, planes=0, colorfamily=vs.GRAY)


def getU(src):
    """ Get U plane from YUV clip """
    return core.std.ShufflePlanes(src, planes=1, colorfamily=vs.GRAY)


def getV(src):
    """ Get V plane from YUV clip """
    return core.std.ShufflePlanes(src, planes=2, colorfamily=vs.GRAY)


def getYUV(src):
    """ Get all planes from a YUV clip in a list """
    return [getY(src), getU(src), getV(src)]


def toYUV(y, u=None, v=None):
    """
    Ugly function but 'y' can optionally be a plane list with all YUV clips,
    or 'y', 'u', and 'v' can simply be a plane for each argument.

    """
    func = 'toYUV'

    if not isinstance(y, list):
        have_u = u is not None
        have_v = v is not None

        if (have_u and not have_v) or (not have_u and have_v) \
            or (have_u and not isinstance(u, vs.VideoNode)) \
            or (have_v and not isinstance(v, vs.VideoNode)):

            raise TypeError(func + ": 'y' must be a YUV plane array or 'y', 'u', and 'v' must be clips")

        y = [y, u, v]

    return core.std.ShufflePlanes(y, planes=[0, 0, 0], colorfamily=vs.YUV)
