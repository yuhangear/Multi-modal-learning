from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class NumElementsBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert check_argument_types()
        assert batch_bins > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        # Sort samples in ascending order 
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")
        if padding:
            # If padding case, the feat-dim must be same over whole corpus,
            # therefore the first sample is referred
            feat_dims = [np.prod(d[keys[0]][1:]) for d in utt2shapes]
        else:
            feat_dims = None

        # Decide batch-sizes
        batch_sizes = []
        current_batch_keys = []
        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if padding:
                for d, s in zip(utt2shapes, shape_files):
                    if tuple(d[key][1:]) != tuple(d[keys[0]][1:]):
                        raise RuntimeError(
                            "If padding=True, the "
                            f"feature dimension must be unified: {s}",
                        )
                bins = sum(
                    len(current_batch_keys) * sh[key][0] * d
                    for sh, d in zip(utt2shapes, feat_dims)
                )
            else:
                bins = sum(
                    np.prod(d[k]) for k in current_batch_keys for d in utt2shapes
                )

            if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        #my batch
        batch_size=100
        batch_items=[]
        index=0

        #总共有31799747/100=317000   2400
        sort_utt_dict=[]
        #??? 3748396/100=37483 85617 2400 
        with open("dump/raw/train_960_text/sort_utt3") as f:

            for line in f:
                line=line.strip()
                sort_utt_dict.append(line)


        for i in sort_utt_dict:
            if index==0:
                temp=[]
           
            temp.append(i)
            if index%batch_size==0 and index!=0:
                
                
                batch_items.append(temp)
                temp=[]
            index+=1

        ###

        #可能会生成2400个文件，最后要匹配13400
        # Set mini-batch
        self.batch_list = []
        #???
        # batch_sizes=batch_sizes*50
        # keys=keys*50
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        index=0
        index_2=0
        #len(keys)=85617
        if keys[0].startswith("test_") != True:
            for key in keys:
                if index%2==0:
                    # self.batch_list.append(tuple( batch_items[index_2%37000] ))
                    index_2=index_2+1
                index=index+1
       


                minibatch_keys.append(key)
                if len(minibatch_keys) == bs:
                    if sort_in_batch == "descending":
                        minibatch_keys.reverse()
                    elif sort_in_batch == "ascending":
                        # Key are already sorted in ascending
                        pass
                    else:
                        raise ValueError(
                            "sort_in_batch must be ascending"
                            f" or descending: {sort_in_batch}"
                        )
                    #if key.startswith("test_") == True:

                    self.batch_list.append(tuple(minibatch_keys))



                    ##???
                    #强行在这，加自己准备的batch_list
                    # if key.startswith("test_") != True:
                        # self.batch_list.append(tuple( batch_items[index%402000] ))
                        # index=index+1
                    #     self.batch_list.append(tuple( batch_items[index%402000] ))
                    #     index=index+1
                    #     self.batch_list.append(tuple( batch_items[index%402000] )) 350000
                    #     index=index+1
                    #     None

                    # if key.startswith("test_") != True:
                    #     self.batch_list.append(tuple( batch_items[index%2799] ))
                    #     index=index+1
                    
                    minibatch_keys = []
                    try:
                        bs = next(iter_bs)
                    except StopIteration:
                        break
        else:


            for key in keys:
                
                minibatch_keys.append(key)
                if len(minibatch_keys) == bs:
                    if sort_in_batch == "descending":
                        minibatch_keys.reverse()
                    elif sort_in_batch == "ascending":
                        # Key are already sorted in ascending
                        pass
                    else:
                        raise ValueError(
                            "sort_in_batch must be ascending"
                            f" or descending: {sort_in_batch}"
                        )
                    #if key.startswith("test_") == True:
                    self.batch_list.append(tuple(minibatch_keys))
                    ##???
                    #强行在这，加自己准备的batch_list
                    # if key.startswith("test_") != True:
                    #     self.batch_list.append(tuple( batch_items[index%402000] ))
                    #     index=index+1
                    #     self.batch_list.append(tuple( batch_items[index%402000] ))
                    #     index=index+1
                    #     self.batch_list.append(tuple( batch_items[index%402000] )) 350000
                    #     index=index+1
                    #     None
                    # if key.startswith("test_") != True:
                    #     self.batch_list.append(tuple( batch_items[index%317000] ))
                    #     index=index+1

                        
                    minibatch_keys = []
                    try:
                        bs = next(iter_bs)
                    except StopIteration:
                        break


        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
