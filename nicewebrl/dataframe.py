from typing import List
import polars as pl
import numpy as np
from flax import struct

class DataFrame(object):

    def __init__(
            self,
            df: pl.DataFrame,
            episodes: List[struct.PyTreeNode],
            index_key: str='index'):
        if "index" not in df.columns:
            df = df.with_row_count("index")
        self._df = df
        self._episodes = episodes
        self._index_key = index_key

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._df, name)

    def filter(self, **kwargs):
        df = self._df.filter(**kwargs)
        idxs = np.array(df[self._index_key])
        episodes = [self._episodes[idx] for idx in idxs]
        return DataFrame(
            df=df,
            episodes=episodes,
            index_key=self._index_key)

    def split_apply(
            self,
            fn,
            split_filter_fn,
            split_filter_settings: dict,
            comp_settings: dict,
            comp_filter_fn=None,
            split_key: str = 'user_id'):

        keys = self._df[split_key].unique()
        array = []
        for key in keys:
            filter_df = self.filter(**split_filter_settings, **{split_key: key})
            if split_filter_fn(filter_df):
                continue
            comp_df = self.filter(**comp_settings, **{split_key: key})
            val = comp_df.apply(fn=fn, filter_fn=comp_filter_fn)
            if len(val) > 0:
                array.append(np.array(val))

        return array

    def filter_row_with_episode(self, filter_fn):
        new_rows = []
        new_episodes = []
        for idx, row in enumerate(self._df.iter_rows(named=True)):
            if not filter_fn(self.episodes[idx]):
                new_rows.append(row)
                new_episodes.append(self._episodes[idx])
        
        new_df = pl.DataFrame(new_rows)
        return DataFrame(
            df=new_df,
            episodes=new_episodes,
            index_key=self._index_key
        )

    def apply(self, fn, filter_fn=None, post_fn = lambda x: x, **kwargs):
        if len(kwargs) > 0:
            eval_df = self.filter(**kwargs)
        else:
            eval_df = self
        array = []
        for e in eval_df.episodes:
            if filter_fn is not None:
                if filter_fn(e): continue
            val = fn(e)

            array.append(val)

        return post_fn(array)

    @property
    def episodes(self):
        return self._episodes

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, key):
        return self._df[key]
    