import torch
from parlai.core.torch_agent import History


class EmpathyHistory(History):
    """
        This class is used for the recording of persona information

    """
    def __init__(
        self,
        opt,
        field='text',
        maxlen=None,
        size=-1,
        p1_token='__p1__',
        p2_token='__p2__',
        dict_agent=None,
    ):
        super().__init__(opt,
        field=field,
        maxlen=maxlen,
        size=size,
        p1_token=p1_token,
        p2_token=p2_token,
        dict_agent=dict_agent)

    def get_history_vec(self):
        """
        Return a vectorized version of the history.
        """
        if len(self.history_vecs) == 0:
            return None

        # vec type is a list
        history = []
        segment = []
        seg_id = 0
        for vec in self.history_vecs[:-1]:
            history += [vec]
            history += [self.delimiter_tok]
            segment += [seg_id] * (len(history[-2])+1)
            if seg_id == 0:
                seg_id = 1
            else:
                seg_id = 0
        history += [self.history_vecs[-1]]
        segment += [seg_id] * len(history[-1])

        if self.temp_history is not None:
            history.extend([self.parse(self.temp_history)])
        if self._global_end_token is not None:
            history += [[self._global_end_token]]

        history = sum(history, [])
        if self.reversed:
            history = list(reversed(history))

        return history, segment








