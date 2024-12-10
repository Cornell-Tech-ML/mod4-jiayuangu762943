import random

import embeddings

import minitorch
from datasets import load_dataset
import numpy as np


BACKEND = minitorch.TensorBackend(minitorch.FastOps)


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)
        # self.bias = RParam(out_channels)
        self.kernel_width =  kernel_width
        self.out_channels = out_channels

    def forward(self, input):
        # TODO: Implement for Task 4.5.
        out = minitorch.conv1d(input, self.weights.value)
        out = out + self.bias.value
        return out

        # batch_size, in_channels, in_length = input.shape
        # out_channels = self.out_channels
        # kernel_width = self.kernel_width

        # # Calculate output length
        # out_length = in_length - kernel_width + 1
        # if out_length <= 0:
        #     raise ValueError(
        #         f"Kernel width {kernel_width} is too large for input length {in_length}."
        #     )

        # # Initialize output tensor with zeros
        # out = input.zeros(
        #     (batch_size, out_channels, out_length))


        # # Perform convolution
        # for b in range(batch_size):
        #     for oc in range(out_channels):
        #         for pos in range(out_length):
        #             conv_sum = 0.0
        #             for inCh in range(in_channels):
        #                 for kw in range(kernel_width):
        #                     conv_sum += input[b, inCh, pos + kw] * self.weights.value[oc, inCh, kw]

        #             conv_sum += self.bias.value[oc]
        #             # Assign to output
        #             out[b, oc, pos] = conv_sum

        # return out


class CNNSentimentKim(minitorch.Module):
    """
    Implement a CNN for Sentiment classification based on Y. Kim 2014.

    This model should implement the following procedure:

    1. Apply a 1d convolution with input_channels=embedding_dim
        feature_map_size=100 output channels and [3, 4, 5]-sized kernels
        followed by a non-linear activation function (the paper uses tanh, we apply a ReLu)
    2. Apply max-over-time across each feature map
    3. Apply a Linear to size C (number of classes) followed by a Dropout with rate 25%
    4. Apply a sigmoid over the class dimension.
    """

    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        # TODO: Implement for Task 4.5.
        self.filter_sizes = filter_sizes
        self.dropout_rate = dropout

        # Initialize Conv1d layers for each filter size
        self.convs = [
            Conv1d(in_channels=embedding_size, out_channels=feature_map_size, kernel_width=fs)
            for fs in filter_sizes
        ]

        # Linear layer: input features = feature_map_size * number of filter sizes
        # Output features = 1 (for binary classification)
        self.linear = Linear(in_size=feature_map_size, out_size=1)


    def forward(self, embeddings, train: bool):
        """
        embeddings tensor: [batch x sentence length x embedding dim]
        """
        # TODO: Implement for Task 4.5.
        transposed_embeddings = embeddings.permute(0, 2, 1)  # Shape: (batch_size, embedding_dim, sentence_length)

    # Pass through each convolutional layer followed by ReLU activation
        conv_output1 = self.convs[0](transposed_embeddings).relu()
        conv_output2 = self.convs[1](transposed_embeddings).relu()
        conv_output3 = self.convs[2](transposed_embeddings).relu()

        # Perform max-over-time pooling on each convolutional output
        pooled1 = minitorch.max(conv_output1, dim=2)  # Shape: (batch_size, feature_map_size)
        pooled2 = minitorch.max(conv_output2, dim=2)
        pooled3 = minitorch.max(conv_output3, dim=2)

        # Combine pooled features by summing them
        combined_pooled = pooled1 + pooled2 + pooled3

        # Flatten the combined features for the fully connected layer
        flattened_features = combined_pooled.contiguous().view(combined_pooled.shape[0], self.feature_map_size)

        # Apply the fully connected layer followed by ReLU activation
        fc_output = self.linear(flattened_features)

        # Apply dropout for regularization (only during training)
        dropped_out = minitorch.dropout(fc_output, self.dropout_rate, ignore = train)

        # Generate output probabilities using sigmoid activation
        output_probabilities = dropped_out.sigmoid().view(transposed_embeddings.shape[0])

        return output_probabilities


# Evaluation helper methods
def get_predictions_array(y_true, model_output, x):
    predictions_array = []
    x = x.to_numpy()
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit, x[j]))
    return predictions_array


def get_accuracy(predictions_array):
    correct = 0
    for y_true, y_pred, logit, xj in predictions_array:
        if y_true == y_pred:
            correct += 1
        # else:
        #     print(y_true, y_pred, xj)
    return correct / len(predictions_array)


best_val = 0.0


def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    best_val = (
        best_val if best_val > validation_accuracy[-1] else validation_accuracy[-1]
    )
    print(f"Epoch {epoch}, loss {train_loss}, train accuracy: {train_accuracy[-1]:.2%}")
    if len(validation_predictions) > 0:
        print(f"Validation accuracy: {validation_accuracy[-1]:.2%}")
        print(f"Best Valid accuracy: {best_val:.2%}")


class SentenceSentimentTrain:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, batch_size)
            ):
                y = minitorch.tensor(
                    y_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + batch_size], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x, train=True)
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -(prob.log() / y.shape[0]).sum()
                loss.view(1).backward()

                # Save train predictions
                train_predictions += get_predictions_array(y, out, x)
                total_loss += loss[0]

                # Update
                optim.step()

            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x, train=False)
                validation_predictions += get_predictions_array(y, out, x)
                validation_accuracy.append(get_accuracy(validation_predictions))
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0


def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in dataset["sentence"][:N]:
        # pad with 0s to max sentence length in order to enable batching
        # TODO: move padding to training code
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                # use random embedding for unks
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)

    # load labels
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    #  Determine max sentence length for padding
    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)

def load_glove_embeddings(path="project/data/glove.6B/glove.6B.50d.txt"):
    """Load GloVe embeddings from local file.

    Args:
    ----
        path: Path to the GloVe embeddings file

    Returns:
    -------
        dict: Word to embedding mapping
    """
    word2emb = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = [float(x) for x in values[1:]]
            word2emb[word] = vector
    return word2emb

def concatenate(tensors, axis=1):
    """
    Concatenate a list of tensors along the specified axis.

    Args:
        tensors (list of Tensor): List of tensors to concatenate.
        axis (int): Axis along which to concatenate.

    Returns:
        Tensor: Concatenated tensor.
    """
    if not tensors:
        raise ValueError("No tensors to concatenate.")

    # Convert all tensors to NumPy arrays
    np_arrays = [t.to_numpy() for t in tensors]

    # Use NumPy to concatenate along the specified axis
    concatenated_np = np.concatenate(np_arrays, axis=axis)

    # Create a new Tensor from the concatenated NumPy array
    concatenated_tensor = minitorch.Tensor.make(
        concatenated_np.flatten(),
        shape=concatenated_np.shape,
        backend=tensors[0].backend
    )

    return concatenated_tensor


if __name__ == "__main__":
    train_size = 450
    validation_size = 100
    learning_rate = 0.005
    max_epochs = 250

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
