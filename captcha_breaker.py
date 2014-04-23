#!/usr/bin/env python

from captcha import *
from skimage import io
import numpy as np

import os, sys

def save_breaker(self, dest):
    import shutil
    
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    
    i = 0
    for cat in self.categories:
        if not cat.name:
            cat.name = '__unnamed_{0}'.format(i)
            i += 1
        path = os.path.join(dest, cat.name)
        if os.path.exists(path):
            print "Skipping duplicit glyph pattern '{0}'".format(cat.name)
            continue
        os.mkdir(path)
        np.save(os.path.join(path, 'pattern_accum.npy'), cat.pattern_accum)
        with file(os.path.join(path, 'sample_count'), 'w') as f:
            f.write('{0}'.format(cat.sample_count))
        io.imsave(os.path.join(path, 'pattern_image.png'), cat.pattern)

def load_breaker(src):
    self = CaptchaBreaker()
    if os.path.exists(src):
        for name in os.listdir(src):
            path = os.path.join(src, name)
            pattern_accum = np.load(os.path.join(path, 'pattern_accum.npy'))
            with file(os.path.join(path, 'sample_count'), 'r') as f:
                sample_count = int(f.read())
            cat = CaptchaBreaker.Category(pattern_accum, sample_count)
            cat.name = name
            self.categories.append(cat)
    return self

def categorize(breaker, samples, max_samples, LEVEL, THRESHOLD):
    processed = 0
    for f in os.listdir(samples):
        if max_samples and processed == max_samples:
            break
        fname = os.path.join(samples, f)
        print "Categorizing '{0}'...".format(fname),
        try:
            sample = io.imread(fname, True)
            breaker.categorize(sample, LEVEL, THRESHOLD)
            processed += 1
            print "OK!"
        except ex:
            print "FAIL!"
            print "--"
            print ex
            print "--"
            pass

def rename(breaker):
    i = 0
    while i < len(breaker.categories):
        io.imshow(breaker.categories[i].pattern, 'pil')
        name = raw_input("Enter new name for {{{0}}}: ".format(breaker.categories[i].name))
        if name != '':
            breaker.categories[i].name = name
            i += 1
        else:
            print "Deleting glyph pattern..."
            del breaker.categories[i]
            

def recognize(breaker, show, sample, LEVEL, THRESHOLD):
    img = io.imread(sample)
    if show:
        io.imshow(img, 'pil')
    res = ''
    for cat in breaker.match(img, LEVEL, THRESHOLD):
        res += cat.name
    print res

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Captcha Breaker')
    parser.add_argument('--datadir', '-d', metavar='DATA', required=True, help='the directory for the glyph patterns')
    subparsers = parser.add_subparsers(help='the subcommand to execute')

    cat_parser = subparsers.add_parser('categorize', help='categorize CAPTCHA samples')
    cat_parser.add_argument('--level', '-l', type=int, default=2, help='maximum size of artifacts to remove when removing the noise')
    cat_parser.add_argument('--threshold', '-t', type=float, default=0.075, help='maximum average pixel difference for "similar" patterns')
    cat_parser.add_argument('--max-samples', '-m', type=int, default=0, dest='max_samples', help='maximum number of samples to process')
    cat_parser.add_argument('samples', metavar='SAMPLES', help='a directory with sample captchas')
    cat_parser.set_defaults(action=lambda breaker, args: categorize(breaker, args.samples, args.max_samples, args.level, args.threshold))

    rename_parser = subparsers.add_parser('rename', help='assign names (characters) to the glyph patterns')
    rename_parser.set_defaults(action=lambda breaker, args: rename(breaker))

    recognize_parser = subparsers.add_parser('recognize', help='recognize a sample (print the decoded CAPTCHA)')
    recognize_parser.add_argument('--show', '-s', action='store_true', help='display the sample being processed')
    recognize_parser.add_argument('--level', '-l', type=int, default=2, help='maximum size of artifacts to remove when removing the noise')
    recognize_parser.add_argument('--threshold', '-t', type=float, default=0.075, help='maximum average pixel difference for "similar" patterns')
    recognize_parser.add_argument('sample', metavar='SAMPLE', help='the CAPTCHA image to recognize (can be an URL)')
    recognize_parser.set_defaults(action=lambda breaker, args: recognize(breaker, args.show, args.sample, args.level, args.threshold))
    
    args = parser.parse_args()
    breaker = load_breaker(args.datadir)
    args.action(breaker, args)
    save_breaker(breaker, args.datadir)
