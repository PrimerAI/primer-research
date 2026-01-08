from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Sample:
    """A sample from the retrieval-response dataset."""

    id: str  # Sample id
    E: str  # Type of embedding model used for retrieval
    Nc: int  # Total number of retrieval candidates
    Np: int  # Total number of positives
    K: int  # The number of retrieved candidates to retain(i.e. the top-K)
    grade: int  # The LLM-assigned grade to the LLM-generate response
    answer_ideal: str  # The response generated from using all the positives documents
    answer_topK: str  # The response generated from using only the top-K documents
    P: float  # The precision of the retrieval
    R: float  # The recall of the retrieval
    rank: list[int]  # Indices of the original documents in ranked order of cosine similarity to the query
    inK: list[Literal[0, 1]]  # Positive/negative flags for the top-K documents


@dataclass
class SegmentData:
    """A subset of samples from the retrieval-response dataset for a specific split."""

    Nc: list[int]  # Total number of retrieval candidates for each sample
    Np: list[int]  # Total number of positives for each sample
    K: list[int]  # The number of retrieved candidates to retain(i.e. the top-K) for each sample
    grade: list[int]  # The LLM-assigned grade to the LLM-generate response for each sample
    inK: list[list[Literal[0, 1]]]  # Positive/negative flags for the top-K documents for each sample
    S2_np: list[int]  # Estimated number of total positives for each sample


@dataclass
class UseCase:
    """A specific configuration for calculating retrieval performance."""

    use_case_code: str

    # The type of measure to caluclate.  One of mF = F measure, mE = F measure with estimated Np,
    # mT = T measure, mNDCG = nDCG measure (which does not have alpha parameter)
    measure: Literal['mF', 'mE', 'mT', 'mNDCG'] = field(init=False)

    # The range over which to vary alpha for the given measure
    alpha_range: np.typing.NDArray = field(init=False)

    def __post_init__(self) -> None:
        """Translate the use case code into a specific measure and range of alpha."""
        usage_config = self.use_case_code.split('^')

        if usage_config[1] == 'x': # No alpha
            alpha_range = np.zeros(1)
        elif usage_config[1][0] == 'l': # log-scale
            n_range, base = int(usage_config[1][1:]), float(usage_config[2])
            alphas = np.logspace(1, n_range, num=n_range, base=base)
            alphasR = 1 - alphas
            alpha_range = np.concatenate((
                np.zeros(1), np.flip(alphas), alphasR, np.ones(1)))
            alpha_range = sorted(alpha_range)
        else: # simple range
            a0, a1, ad = float(usage_config[1]), float(usage_config[2]), float(usage_config[3])
            alpha_range = np.arange(a0, a1, ad)
        
        self.measure = usage_config[0]
        self.alpha_range = alpha_range

        if (self.measure == 'mDCG') and (self.alpha_range != np.zeros(1)).all():
            raise ValueError(
                'nDCG does not have an alpha parameter, yet one is specified.'
            )
