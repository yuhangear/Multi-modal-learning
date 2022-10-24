# from pathlib import Path
# from typing import Dict
# from typing import Iterable
# from typing import List
# from typing import Union

# import numpy as np
# from typeguard import check_argument_types


# class TokenIDConverter:
#     def __init__(
#         self,
#         token_list: Union[Path, str, Iterable[str]],
#         unk_symbol: str = "<unk>",
#     ):
#         assert check_argument_types()

#         if isinstance(token_list, (Path, str)):
#             token_list = Path(token_list)
#             self.token_list_repr = str(token_list)
#             self.token_list: List[str] = []

#             with token_list.open("r", encoding="utf-8") as f:
#                 for idx, line in enumerate(f):
#                     line = line.rstrip()
#                     self.token_list.append(line)

#         else:
#             self.token_list: List[str] = list(token_list)
#             self.token_list_repr = ""
#             for i, t in enumerate(self.token_list):
#                 if i == 3:
#                     break
#                 self.token_list_repr += f"{t}, "
#             self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

#         self.token2id: Dict[str, int] = {}
#         for i, t in enumerate(self.token_list):
#             if t in self.token2id:
#                 raise RuntimeError(f'Symbol "{t}" is duplicated')
#             self.token2id[t] = i

#         self.unk_symbol = unk_symbol
#         if self.unk_symbol not in self.token2id:
#             raise RuntimeError(
#                 f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
#             )
#         self.unk_id = self.token2id[self.unk_symbol]

#     def get_num_vocabulary_size(self) -> int:
#         return len(self.token_list)

#     def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
#         if isinstance(integers, np.ndarray) and integers.ndim != 1:
#             raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
#         return [self.token_list[i] for i in integers]

#     def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
#         return [self.token2id.get(i, 1) for i in tokens]





# class TokenIDConverter_phone:
#     def __init__(
#         self,
     
#     ):

#         self.phonedict={}
#         self.id2token={}
#         self.phonedict["<blank>"]=0
#         self.phonedict["<unk>"]=1
#         index=1
#         with open("dump/raw/all_text_phone",encoding="utf8") as f:
#             for i in f:
#                 i=i.strip()
#                 i=i.split(" ")[1:]
#                 for p  in i:
#                     if( p  not in self.phonedict):
#                         index=index+1
#                         self.phonedict[p]=index

#         for i, t in self.phonedict.items():

#             self.id2token[t] = i


#     def get_num_vocabulary_size(self) -> int:
#         return len(self.phonedict)

#     def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:

#         return [self.id2token[i] for i in integers]

#     def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
#         return [self.phonedict[i] for i in tokens]



from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

import numpy as np
from typeguard import check_argument_types


class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
    ):
        assert check_argument_types()

        if isinstance(token_list, (Path, str)):
            token_list = Path(token_list)
            self.token_list_repr = str(token_list)
            self.token_list: List[str] = []

            with token_list.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    self.token_list.append(line)

        else:
            self.token_list: List[str] = list(token_list)
            self.token_list_repr = ""
            for i, t in enumerate(self.token_list):
                if i == 3:
                    break
                self.token_list_repr += f"{t}, "
            self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

        self.token2id: Dict[str, int] = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, 1) for i in tokens]





class TokenIDConverter_phone:
    def __init__(
        self,
     
    ):

        self.phonedict={}
        self.id2token={}
        self.phonedict["<blank>"]=0
        self.phonedict["<unk>"]=1
        index=1
        with open("dump/raw/all_phone",encoding="utf8") as f:
        
            for i in f:
                i=i.strip()

                index=index+1
                self.phonedict[i]=index

        for i, t in self.phonedict.items():

            self.id2token[t] = i


    def get_num_vocabulary_size(self) -> int:
        return len(self.phonedict)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:

        return [self.id2token[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        temp=[]
        for i in tokens:
            if i in self.phonedict:
                temp.append(self.phonedict[i])
            else:
                temp.append(1)

        #return [ self.phonedict[i] for i in tokens]
        return temp
