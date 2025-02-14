

from abc import ABC, abstractmethod


class ModelBuilder(ABC):

    def __init__(self, model_type, model_complexity):
        super().__init__()
        self.model_type = model_type
        self.model_complexity = model_complexity

    def build(self):

        if self.model_type == "mlp":
            return self._mlp_simple() if self.model_complexity == "simple" else self._mlp_complex()

        elif self.model_type == "cnn":
            return self._cnn_simple() if self.model_complexity == "simple" else self._cnn_complex()

        elif self.model_type == "lstm":
            return self._rnn_simple() if self.model_complexity == "simple" else self._rnn_complex()


    @abstractmethod
    def _mlp_simple(self):
        pass

    @abstractmethod
    def _mlp_complex(self):
        pass

    @abstractmethod
    def _cnn_simple(self):
        pass

    @abstractmethod
    def _cnn_complex(self):
        pass

    @abstractmethod
    def _rnn_simple(self):
        pass

    @abstractmethod
    def _rnn_complex(self):
        pass