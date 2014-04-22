import numpy as np
from itertools import chain

""" Converts an 8-bit grayscale image into a numpy array of booleans (False = white, True = black).
    All non-white pixels are interpreted as black. """
def to_bitmap(image):
    return (image / 255).astype(np.bool)

def transform_bitmap(bitmap, transform):
    a = np.fromiter(chain.from_iterable(transform(bitmap)), np.bool)
    a.shape = bitmap.shape
    return a

""" Repaints all sequences of black bits shorter than or equal to 'level' to white. """
def flatten_line(line, level):
    black = 0
    for i in line:
        if i:
            c = 0 if black > level else 1
            for _ in xrange(black):
                yield c
            black = 0
            yield 1
        else:
            black += 1
    
    c = 0 if black > level else 1
    for _ in xrange(black):
        yield c

""" Applies 'flatten_line' to all lines in the bitmap. """
def flatten_lines(bitmap, level):
    for line in bitmap:
        yield flatten_line(line, level)

""" Transforms 'bitmap' by applying 'flatten_lines' to rows. """
def flatten_h(bitmap, level):
    return transform_bitmap(bitmap, lambda b: flatten_lines(b, level))

""" Transforms 'bitmap' by applying 'flatten_lines' to columns. """
def flatten_v(bitmap, level):
    return transform_bitmap(bitmap.T, lambda b: flatten_lines(b, level)).T

""" Extracts all separate contigous black areas into separate cropped bitmaps. """
def flood_split(bitmap):
    bitmap = bitmap.copy()
    sr = bitmap.shape[0]
    sc = bitmap.shape[1]
    
    def flood_extract(r0, c0):
        result = np.zeroes_like(bitmap)
        stack = [(r0, c0)]
        bitmap[r0, c0] = 0
        min_r = r0
        max_r = r0
        min_c = c0
        max_c = c0
        while len(stack) > 0:
            r, c = stack.pop()
            result[r, c] = 1
            if r - 1 >= 0 and bitmap[r - 1, c]:
                if r is min_r:
                    min_r = r - 1
                bitmap[r - 1, c] = 0
                stack.append((r - 1, c))
            if r + 1 < sr and bitmap[r + 1, c]:
                if r is max_r:
                    max_r = r + 1
                bitmap[r + 1, c] = 0
                stack.append((r + 1, c))
            if c - 1 >= 0 and bitmap[r, c - 1]:
                if c is min_c:
                    min_c = c - 1
                bitmap[r, c - 1] = 0
                stack.append((r, c - 1))
            if c + 1 < sc and bitmap[r, c + 1]:
                if c is max_c:
                    max_c = c + 1
                bitmap[r, c + 1] = 0
                stack.append((r, c + 1))
        return result[min_r : max_r + 1, min_c : max_c + 1]
    
    for r in xrange(sr):
        for c in xrange(sc):
            if bitmap[r, c]:
                yield flood_extract(r, c)

""" Removes noise smaller than or equal to 'level' from the bitmap. """
def flatten(bitmap, level):
    while True:
        new_bitmap = flatten_h(bitmap, level) | flatten_v(bitmap, level)
        if (new_bitmap == bitmap).all():
            break
        bitmap = new_bitmap
    return bitmap

""" Computes absolute difference of two bitmaps. """
def diff(img1, img2):
    return np.absolute(img1.astype(np.float64) - img2.astype(np.float64))

""" Computes average absolute difference of two bitmaps (their "non-similarity"). """
def diff_value(img1, img2):
    return diff(img1, img2).mean()

def match(i1, i2, threshold):
    rs = min(i1.shape[0], i2.shape[0])
    cs = min(i1.shape[1], i2.shape[1])
    for r1 in xrange(i1.shape[0] - rs + 1):
        for r2 in xrange(i2.shape[0] - rs + 1):
            for c1 in xrange(i1.shape[1] - cs + 1):
                for c2 in xrange(i2.shape[1] - cs + 1):
                    s1 = i1[r1:r1+rs, c1:c1+cs]
                    s2 = i2[r2:r2+rs, c2:c2+cs]
                    if diff_value(s1, s2) < threshold:
                        yield (r1, c1), (r2, c2), (rs, cs)

class CaptchaBreaker(object):
    class Category(object):
        def __init__(self, pattern_accum, sample_count=1):
            self.name = None
            self.pattern_accum = pattern_accum.astype(np.int32)
            self.sample_count = sample_count
            if sample_count is 1:
                self.pattern = pattern_accum.astype(np.float64)
            else:
                self.pattern = np.divide(pattern_accum, float(sample_count))
        
        def __repr__(self):
            return "Category '{0}' [{1} samples]".format(self.name, self.sample_count)
    
    def __init__(self):
        self.categories = []

    def categorize(self, captcha, level, threshold):
        for letter in flood_split(flatten(to_bitmap(captcha), level)):
            new_cat = CaptchaBreaker.Category(letter)
            while new_cat is not None:
                m = None
                for i in xrange(len(self.categories)):
                    cat = self.categories[i]
                    m = next(match(new_cat.pattern, cat.pattern, threshold), None)
                    if m is not None:
                        is1, is2, s = m
                        del self.categories[i]
                        new_cat = CaptchaBreaker.Category(
                            new_cat.pattern_accum[is1[0]:is1[0]+s[0], is1[1]:is1[1]+s[1]] + 
                            cat.pattern_accum[is2[0]:is2[0]+s[0], is2[1]:is2[1]+s[1]],
                            new_cat.sample_count + cat.sample_count)
                        break
                if m is None:
                    self.categories.append(new_cat)
                    new_cat = None
                    break
                
    def match(self, captcha, level, threshold):
        sample = flatten(to_bitmap(captcha), level).astype(np.float64)
        res = {}
        for cat in self.categories:
            for m in match(sample, cat.pattern, threshold):
                res[m[0][1]] = cat
        prev_column = -level - 1
        for column in sorted(res.iterkeys()):
            if column - prev_column <= level:
                continue
            yield res[column]
            prev_column = column

    def __repr__(self):
        return repr(self.categories)

__all__ = ('to_bitmap', 'flood_split', 'flatten', 'diff', 'diff_value', 'CaptchaBreaker')