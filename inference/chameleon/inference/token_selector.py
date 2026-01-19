import torch

SPEECH_START = 12821  # <boa>
SPEECH_END   = 12820  # <eoa>
INS_TOKEN    = 12840  # <-Ins->
RES_TOKEN    = 12841  # <-Res->


class TokenSelector:
    def __call__(
        self, input_ids: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids.shape=[batch, seq_len]
        # probs.shape=[batch, vocab]
        ...


class ArgmaxTokenSelector(TokenSelector):
    def __call__(
        self, _: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.LongTensor:
        # probs.shape=[batch, vocab]
        return probs.argmax(dim=1)


class MultinomialTokenSelector(TokenSelector):
    def __call__(
        self, _: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.LongTensor:
        # probs.shape=[batch, vocab]
        return probs.multinomial(num_samples=1).squeeze(1)


class ReplicatedInputTokenSelector(TokenSelector):
    def __init__(self, token_selector: TokenSelector, n: int):
        self.token_selector = token_selector
        self.n = n

    def __call__(
        self, input_ids: torch.LongTensor, probs: torch.FloatTensor
    ) -> torch.LongTensor:
        # input_ids.shape=[n*batch, seq_len]
        # probs.shape=[n*batch, vocab]
        primary_input_ids = torch.chunk(input_ids, chunks=self.n, dim=0)[0]
        primary_probs = torch.chunk(probs, chunks=self.n, dim=0)[0]
        tokens = self.token_selector(primary_input_ids, primary_probs)
        return tokens.repeat(self.n)
    
class FiniteTransTokenSelector(TokenSelector):
    """
      - prev in {INS_TOKEN, SPEECH_START}  => greedy = True
      - prev in {RES_TOKEN, SPEECH_END}    => greedy = False
    """
    def reset(self):
        self._mode = None
        
    def __init__(self, default_greedy: bool = False):
        self.default_greedy = default_greedy
        self._mode: torch.BoolTensor | None = None  # [batch]

    def __call__(self, input_ids: torch.LongTensor, probs: torch.FloatTensor) -> torch.LongTensor:
        # input_ids: [batch, seq_len], probs: [batch, vocab]
        prev = input_ids[:, -1]
        bsz = prev.shape[0]

        if self._mode is None or self._mode.shape[0] != bsz or self._mode.device != prev.device:
            self._mode = torch.full((bsz,), self.default_greedy, device=prev.device, dtype=torch.bool)

        start_mask = (prev == INS_TOKEN) | (prev == SPEECH_START)
        end_mask   = (prev == RES_TOKEN) | (prev == SPEECH_END)

        self._mode |= start_mask
        self._mode &= ~end_mask

        greedy_ids = probs.argmax(dim=1)

        if self._mode.all():
            return greedy_ids
        
        sample_ids = probs.multinomial(num_samples=1).squeeze(1)

        return torch.where(self._mode, greedy_ids, sample_ids)
