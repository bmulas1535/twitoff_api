import numpy as np
import re


class BinaryCrossEntropy(object):
  def __init__(self): pass

  def loss(self, y, p):
    # If p == 1, division by zero occurs
    p = np.clip(p, 1e-15, 1-1e-15)
    loss = (y * np.log(p) - (1 - y) * (np.log(1 -p)))
    return -loss

  def gradient(self, y, p):
    # Avoid that pesky division by zero problem
    p = np.clip(p, 1e-15, 1-1e-15)
    grad = (-y / p) + ((1 - y) / (1 - p))
    return grad


class Sigmoid(object):
  def __call__(self, s):
    return 1 / (1 + np.exp(-s))

  def gradient(self, s):
    return self.__call__(s) * (1 - self.__call__(s))


def clean_text(status) -> str:
  """Removes artifacts from status text that do not
  impact the contextual substance of the status update.

  Args:
    status (tweepy.models.Status): A status update retrieved using the
    Tweepy module.

  Returns:
    A string representation of the status update.
  """

  status_text = status.text.encode("ascii", "ignore").decode()
  res = re.sub(r"@\w*", "", status_text)
  res = re.sub(r"&\w*;", "", res)
  res = re.sub(r"#\w*", "", res)
  res = re.sub(r"\s+", " ", res)
  res = re.sub(r"http\w?://.*", "", res)
  res = re.sub(r"([a-zA-Z]+)[\'`]([a-zA-Z]+)", r"\1"r"\2", res)
  res = re.sub(r"[\.\,\!\?\\\/]", "", res).strip()
  return res.lower()

def create_data(X, y):
  "Creates the corpus."""
  corpus = np.array([clean_text(x) for x in X])
  to_remove = list()
  for i, element in enumerate(corpus):
    if len(element) <= 5:
      to_remove.append(i)

  X = np.delete(corpus, to_remove)
  y = np.delete(y, to_remove).reshape(-1, 1)
  return X, y
