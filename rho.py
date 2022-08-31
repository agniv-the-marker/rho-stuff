import pandas as pd
import numpy as np
from math import floor, ceil, sqrt, log
from decimal import *
import matplotlib.pyplot as plt

source = "./drive/MyDrive/Figures/rho.csv"

def resetRho():
  """ Empties out if error is encountered """
  df = pd.DataFrame({'val':[],'out':[],'precision':[]})
  df.to_csv(source, index=False)
  return df
  
def getPrec(k, df):
  """ Returns the stored precision for rho(k) in the csv. 0 if not found """
  try:
    return df.loc[min(df.loc[df.val == k].index)].precision
  except:
    return 0

def addRho(df, info):
  """ Adds value to the stored csv. """
  inside = getPrec(info[0], df) != 0
  if inside:
    df.loc[(df["val"] == info[0])] = info
  else:
    df.loc[len(df)] = info
  return df
 
def writeRho(df):
  """ Writes to an external csv for storage """
  df.to_csv(source, index = False)

def left_gen(k):
  """ Generator for leftmost nodes """
  yield 0
  l = 1
  while True:
    yield l
    l = ceil(Decimal(l) * Decimal(k)) 

def row_gen(k):
  """ Generator for row lengths """
  left_iter = left_gen(k)
  last_left = next(left_iter)
  while True:
    next_left = next(left_iter)
    yield next_left - last_left
    last_left = next_left

def get_rho(k, min_row_size = 10**3, eps = 10 ** -3, iter_upper_bound = 10 ** 2):
  """
  If a rho(k) value is not found within rho.csv, generate it. 
  Also runs if a higher precision value 
  precision is defined with:
  minimum row size to start at, upper bound to stop at, eps, decimal precision
  """
  if k <= 1:
    return None
  elif abs(k - int(k)) < 10**(-1000000):
    return Decimal((k-1))/Decimal(k)
  row = row_gen(k)
  last_c = 1
  r = next(row)
  c = 1
  while r < min_row_size:
    r = next(row)
    c += 1
  for i in range(c, iter_upper_bound+c):
    next_c = Decimal(next(row)) / Decimal(k) ** Decimal(i)
    if abs(next_c - last_c) < eps:
      return (next_c + last_c) / 2
    last_c = next_c
  return next_c

def find_rho(k, min_row_size = 10**3, eps = 10 ** -3, iter_upper_bound = 10 ** 2):
  """
  Gets the value of rho(k) to some precision, either through rho.csv or generating it.
  """
  try:
    df = pd.read_csv(source)
  except FileNotFoundError:
    return get_rho(k, min_row_size=min_row_size, eps=eps, iter_upper_bound=iter_upper_bound)
  except:
    df = resetRho()
  if abs(k - int(k)) < 10**(-100):
    prec = np.inf
  else:
    prec = log(min_row_size*eps**(-1)*iter_upper_bound*getcontext().prec, 10) 
  curPrec = getPrec(k, df)
  if curPrec < prec or curPrec == 0:
    rho = get_rho(k, min_row_size=min_row_size, eps=eps, iter_upper_bound=iter_upper_bound)
    writeRho(addRho(df, [k, rho, prec]))
    return rho
  elif curPrec >= prec:
    return df.loc[min(df.loc[df.val == k].index)].out

x = np.linspace(1 + 10**(-6), 6, 10**5)
plt.scatter(x, np.array(list(map(find_rho, x))), s = 0.1)
plt.xlabel('k')
plt.ylabel('ρ(k)')
plt.title(f'Estimated values of ρ(k) for 100000 random values 1 < k < 6')
plt.show()
