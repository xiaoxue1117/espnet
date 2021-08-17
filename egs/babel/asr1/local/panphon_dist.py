#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import panphon

ft = panphon.featuretable.FeatureTable()

def dist(a, b):
    seg_a = ft.word_fts(a)[0]
    print(a, seg_a)
    try:
        seg_b = ft.word_fts(b)[0]
        print(b, seg_b)
    except:
        import pdb;pdb.set_trace()
    return seg_a.distance(seg_b)

if __name__ == "__main__":
    a='o'
    b='ʏ'
    print(dist(a, b))
