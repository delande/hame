"""
This module computes one-photon and two-photon matrix elements between hydrogenic states.

It includes diagonal elements of the two-photon matrix elements, that is light-shift and one-photon ionization rate.

It performs the numerical calculation in the length gauge in a Sturmian basis.
It additionally performs the calculation in the velocity gauge.

It checks that results in length and velocity gauges are equal.
"""

__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2023 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.2"

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
# ____________________________________________________________________
#
# hame.py
# Author: Dominique Delande
# Release date: February, 24, 2023

import numpy as np
import math
import cmath
import scipy.special
import sys
import mpmath

def compute_dilatation_matrix(two_k: int, gamma: complex, nsup: int, debug: bool=False) -> np.ndarray:
  r"""
  Compute the dilatation matrix for a :math:`D_k^+` irreducible representation of the SO(2,1) group.

  The dilatation operator is defined as :math:`\exp(-i\gamma S_2)` where S_2 is the second generator of the SO(2,1) group
  and \gamma the parameter of the dilatation.
  Here, we compute the matrix of this operator in the eigenbasis of the S_3 generator of the SO(2,1), for a D_k^+ irreproducible representation
  The eigenbasis of S_3 is such that S_3|n> = (n+k)|n>, see eq. (I-168) of [1]_, where n is a non-negative integer.
  k is the parameter of the D_k^+ representation and is a positive half-integer or integer.
  The matrix elements are given in Eq. (I-210) of [1]_. They involve hypergeometric functions.
  However, for small non-zero \gamma, this equation generates underflows and overflows and is thus inconvenient.
  A better strategy, used in this routine, is to use recurrence relations between matrix elements to generate all of them
  from the ones at small n,n'. One must be careful to use stable recurrence relations.

  Parameters
  ----------
  two_k : int
    Twice the k parameter of the irreducible parameter. As k is either an half-integer of an integer, two_k is an integer.
    For the 3D hydrogen atom, k is l+1, so that two_k = 2*l+2 should be used. In that case, the quantum number n
    is n_{principal}-l-1, when n_{principal} is the usual principal quantum number.
  gamma : complex
    The parameter of the dilatation matrix. It can be either a float or a complex. For a standard dilatation, \gamma is real
    and the dilatation matrix is purely real. The complex \gamma case is used when a complex dilatation takes place. This is used
    when complex rotation of the coordinates is used. This is not documented in [1]_.
  nsup : int
    The maximum n involved in the calculation.
  debug : bool, optional
    The default is False.
    The dilatation matrix is orthogonal so that O*Ot should be unity. When the matrix is truncated, this is no longer true.
    When debug is True, the routine computes O*Ot restricted to the [0:nmax,0:nmax] subspace (for nmax from 1 to nsup)
    and prints how much it differs from the unit matrix.
    Note that, for complex \gamma, the dilatation matrix is complex, but still orthogonal (NOT unitary).

  Returns
  -------
  A numpy array :math:`[0\!:\!nsup\!+\!1,0\!:\!nsup\!+\!1]` containing :math:`<\!n k|\exp(-i\gamma S_2)|n' k\!>` for 0<=n,n'<=nsup.

  References
  ----------
  .. [1] D. Delande, These d'Etat: Atomes de Rydberg en champs statiques intenses, Universite Pierre et Marie Curie (Paris), 1988, https://theses.hal.science/tel-00011864v1

  """
  assert isinstance(two_k,int), 'Inside compute_dilatation_operator: two_k should be an integer and is '+str(two_k)
  assert isinstance(nsup,int), 'Inside compute_dilatation_operator: nsup  should be an integer and is '+str(nsup)
  if type(gamma)==complex:
    my_type = complex
  else:
    my_type = float
 # First the simple case where teta is unity, so that the matrix is the unit matrix
  if cmath.isclose(gamma,0.0):
    return np.identity(nsup+1)
# The generic case
  x = np.zeros((nsup+1,nsup+1),dtype=my_type)
# Determine (real or complex) cosh(\gamma/2) and sinh(\gamma/2)
  if type(gamma)==complex:
    ch = cmath.cosh(0.5*gamma)
    sh = cmath.sinh(0.5*gamma)
  else:
    ch = math.cosh(0.5*gamma)
    sh = math.sinh(0.5*gamma)
  th = sh/ch
  invth = ch/sh
# the simplest matrix element
  x[0,0] = ch**(-two_k)
# recurrence relation for n=0 and nc from 1 to nsup
  for n in range(nsup):
    x[0,n+1] = x[0,n]*th*math.sqrt((n+two_k)/(n+1))
# a is a vector of suitable length alternating -1 and +1 : [-1,1,-1,1,...]
  a = np.tile(np.array([-1,1]),(nsup+1)//2)
# Symetrisation with proper sign
  x[1:nsup+1,0] = x[0,1:nsup+1]*a[0:nsup]
# Compute now the next rows/columns
  for nc in range(1,nsup+1):
# Compute column nc
# The second element in the column
    x[1,nc] = (-((nc+two_k)*th-nc*invth)*x[0,nc])/math.sqrt(two_k)
# For elements up to n=nc, the recurrence relation is stable
    for n in range(1,nc):
      x[n+1,nc] = (-((n+nc+two_k)*th+(n-nc)*invth)*x[n,nc] - math.sqrt(n*(n+two_k-1))*x[n-1,nc])/math.sqrt((n+1)*(n+two_k))
# Symmetrize with the proper sign
    x[nc,nc-1:0:-1] = x[nc-1:0:-1,nc] * a[0:nc-1]
  if debug:
    print('Inside compute_dilatation_matrix: two_k = ',two_k,' gamma = ',gamma,' nsup = ',nsup)
    y = x@np.transpose(x) - np.identity(nsup+1,dtype=type(gamma))
    for n in range(1,nsup+1):
      print('nmax = ',n,' residual norm of O*Ot-1 = ',np.linalg.norm(y[0:n,0:n]))
  return x

def gordon_formula(n: int, l: int, m: int, nprime: int, lprime: int) -> float:
  """
  Compute the matrix element of the dipole operator between eigenstates of the hydrogen atom.

  This is <n l m | z | nprime, lprime, m>.
  For n=nprime, it is given by Eq. (I-48) in [1]_.
  For n!=nprime, it is computed using the so-called Gordon formula, Eq. (I-49) of ref. [1]_, It was originally given by Eq. (28) of ref [2]_
  and is also reproduced in Eq. (63.2)  of ref. [3]_.
  Note that the angular part, Eq. (I-45) of [1]_ is included,
  so that it is actually the matrix element of the z operator.
  When matrix elements of the x and y operators are needed, they can be straightforwardly deduced from the ones of z,
  using Eqs. (I-46,I-47) of [1]_.


  Parameters
  ----------
  n : int
    Principal quantum number of the first state.
  l : int
    Angular momentum of the first state.
  m: int
    Magneiic quantum number of the two states.
  nprime : int
    Principal quantum number of the second state.
  lprime : int
    Angular momentum of the second state.

  Returns
  -------
  float
    The matrix element <n l m | z | nprime, lprime, m>

  Notes
  -----
  When the principal quantum numbers n,nprime are large, the factorials in the Gordon formula may become huge and overflow.
  A slight rewriting using scipy.special.comb (the number of combinations, using ratio of factorials) solves this issue.
  Note that this routine has been tested for small l,lprime values and may overflow/underflow for large values of l,lprime (not tested).

  References
  ----------
  .. [1] D. Delande, These d'Etat: Atomes de Rydberg en champs statiques intenses, Universite Pierre et Marie Curie (Paris), 1988, https://theses.hal.science/tel-00011864v1
  .. [2] W. Gordon, Ann. Phys 2, 1031 (1929)
  .. [3] H.A. Bethe and E. E. Salpeter, Quantum Mechanics of One- and Two-Electron Atoms, Springer (1957)
  """
# The trivial case where |Delta(l)| is not 1
  if abs(l-lprime) != 1:
    return 0.0
# make sure that lprime is l-1
  if l<lprime:
    lprime, l = l,lprime
    nprime, n = n, nprime
  if abs(m)>=l:
    return 0
# The n=nprime case is given by Eq. (I-48) in [1]
  if n==nprime:
    return -1.5*n*math.sqrt((n-l)*(n+l))*math.sqrt((l**2-m**2)/(4*l**2-1))
  x = scipy.special.hyp2f1(-nprime+l,-n+l+1,2*l,-4*n*nprime/((n-nprime)**2)) - (((n-nprime)/(n+nprime))**2)*scipy.special.hyp2f1(-nprime+l,-n+l-1,2*l,-4*n*nprime/((n-nprime)**2))
#  print(x)
  x *= ((4*n*nprime)/((n-nprime)**2))**(l+1)*(((n-nprime)/(n+nprime))**(n+nprime))
#  print(x)
# the next factor combines the square root of ratio/products of factorials and the (2*l-1)! in the denominator.
  x *= math.sqrt(2*l*(2*l+1)*scipy.special.comb(n+l,2*l+1)*scipy.special.comb(nprime+l-1,2*l-1))
#  print(x)
# the rightmost term is the angular part (from "r" to "z")
  x *= 0.25*((-1)**(nprime-l))*math.sqrt((l**2-m**2)/(4*l**2-1))
  return x

def ionization_rate_1s(omega: float) -> float:
  """
  Compute the ionization rate of the 1s state.

  It uses the Gordon formula discrete->continuum, Eq. (28') in ref. [1]_. Note however that there are two typos in the equation:

  1. The factor i in the first numerator should not be in the exponent. but a simple multiplication factor

  2. In the first denominator, the power of x should be 2, not 3/2.

  When specialized to the 1s initial state, the formula simplifies to Eq. (71.4) in ref. [2]_.

  Note that the result should be identical to the one of hame.ionization_rate(1, 0, 1, omega).

  Parameters
  ----------
  omega : float
    photon frequency

  Returns
  -------
  float
    ionization rate

  References
  ----------
  .. [1] W. Gordon, Ann. Phys 2, 1031 (1929)
  .. [2] H.A. Bethe and E. E. Salpeter, Quantum Mechanics of One- and Two-Electron Atoms, Springer (1957)

  """
  if omega<=0.5:
    return 0.0
  w = omega-0.5
  n = 1.0/np.sqrt(2.0*w)
  rate = 2.0*np.pi*256.*(n**2/(1.0+n**2))**5*np.exp(-4.0*n*np.arctan(1.0/n))/(3.0*(1.0-np.exp(-2.0*np.pi*n)))
  return rate

def ionization_rate(nprime: int, lprime: int, m: int, l: int, omega: float, debug: bool=False) -> float:
  """
  Compute the ionization rate of the (nprime,lprime) state to the l continuum.

  It uses the Gordon formula discrete->continuum, Eq. (28') in ref. [1]_. Note however that there are two typos in the equation:

  1. The factor i in the first numerator should not be in the exponent, but a simple multiplication factor

  2. In the first denominator, the power of x should be 2, not 3/2

  Parameters
  ----------
  nprime : int
    principal quantum number of the initial state
  lprime: int
    angular momentum of the initial state
  m: int
    magnetic quantum number of the initial and final states
  l: int
    angular momentum of the final state

  omega : float
    photon frequency
  debug: boolean
    it True, print some intermediate values

  Returns
  -------
  float
    ionization rate

  References
  ----------
  .. [1] W. Gordon, Ann. Phys 2, 1031 (1929)
  """
  if abs(l-lprime) != 1:
    return 0.0
  if omega<=0.5/(nprime**2):
    return 0.0
  w = omega-0.5/(nprime**2)
  n = 1.0/np.sqrt(2.0*w)
# The original formula of Gordon is rather complicated. Significantly simpler results are obtained by the simple substitution n->1j*n
# A small price to pay is to drop the phase factors, irrelevant as long as we compute the ionization rate
  if l == lprime+1:
# This is Eq. (28') of Gordon
#    kprime = 1/nprime
    u = (1j*n-nprime)/(1j*n+nprime)
# For complex values of arguments/parameters of hypergeometric functions, the mpmath package is needed
# Note, however, that all hypergeometric functions boil down to polynomials, so that a faster implementation could be used
    x1 = mpmath.hyp2f1(l+1-1j*n,lprime+1-nprime,2*l,-4j*n*nprime/((1j*n-nprime)**2))
    x2 = mpmath.hyp2f1(l-1-1j*n,lprime+1-nprime,2*l,-4j*n*nprime/((1j*n-nprime)**2))
# As u is always of modulus 1, terms u**(real_power) are only phase factors and dropped
    r = (x1-u**2*x2)*(u**(1j*n))
    r *= 0.25*np.sqrt(scipy.special.factorial(nprime+lprime)/scipy.special.factorial(nprime-lprime-1))/scipy.special.factorial(2*l-1)
#    r *= n**2.0*(4.0*kprime/(n*(kprime**2+1.0/n**2)))**(l+1)
#    r *= n**2.0*(4*n*nprime/(n**2+nprime**2))**(l+1)
# In the following equation, one sqrt(n) comes from the sqrt of the factorials
# An additional n**1.5 comes from the normalization (to be checked...)
    r *= n**2.0*(4j*n*nprime/((1j*n-nprime)**2))**(l+1)
#    *np.exp(-2.0*n*np.arctan(1.0/(n*kprime)))
#    print(np.exp(-2.0*n*np.arctan(1.0/(n*kprime))))
#    print(((1j*n-nprime)/(1j*n+nprime))**(1j*n))
    for s in range(1,l+1):
      r *= np.sqrt(s**2+n**2)
# In Eq. (28') of Gordon, the first part of the exponential combines with the sinh function to simply give:
    r /= np.sqrt(1.0-np.exp(-2.0*np.pi*n))
# r is the radial matrix element
# To obtain the ionization rate, the angular part has to be taken into account
# The ionization rate is given by the Fermi golden rule.
# The normalization of the continuum wavefunction is such that the density of states is unity
    rate = 2.0*np.pi*abs(r)**2*(l**2-m**2)/(4*l**2-1)
  if l == lprime-1:
# Gordon does not give any equation for this case
# The simplest is to start from the discrete-discrete matix element and make the change n->1j*n
    u = (1j*n-nprime)/(1j*n+nprime)
    x1 = mpmath.hyp2f1(lprime+1-nprime,lprime-1j*n,2*lprime,-4j*n*nprime/((nprime-1j*n)**2))
    x2 = mpmath.hyp2f1(lprime-1-nprime,lprime-1j*n,2*lprime,-4j*n*nprime/((nprime-1j*n)**2))
    r = (x1-u**2*x2)*(u**(1j*n))
    if debug:
# The following line is because of the strange types used by mpmath
      r = complex(r)
      print('n,l',n,l)
      print('nprime,lprime',nprime,lprime)
      print('u',u,'arg',np.angle(u))
      print('r',r,'arg',np.angle(r))
    r *= 0.25*np.sqrt(scipy.special.factorial(nprime+lprime)/scipy.special.factorial(nprime-lprime-1))/scipy.special.factorial(2*lprime-1)
    r *= n**2.0*(4j*n*nprime/((1j*n-nprime)**2))**(lprime+1)
    for s in range(1,lprime):
      r *= np.sqrt(s**2+n**2)
    r /= np.sqrt(1.0-np.exp(-2.0*np.pi*n))
    rate = 2.0*np.pi*abs(r)**2*(lprime**2-m**2)/(4*lprime**2-1)
  return rate

def compute_2r_matrix(l: int, nsup: int, alp: complex) -> np.ndarray:
  r"""
  Compute the matrix representing the 2r operator in a Sturmian basis of parameter alp.

  This matrix is tridiagonal.
  The 2r operator in the Sturmian basis is, according to Eq. (I-211) of [1], 2r = 2*alp(U_3+U_1) = 2*alp*(S_3+S_1+T_3+T_1)
  Its matrix elements are given by Eq. (I-197) of [1].

  Parameters
  ----------
  l : int
    Angular momentum quantum number.
  nsup : int
    Largest value of the principal quantum number.
  alp : complex or float
    Dilatation parameter

  Returns
  -------
  Numpy array : complex or float
    It should be a tridiagonal matrix :math:`[0\!:\!nsup\!-\!l,0\!:\!nsup\!-\!l]`.

  References
  ----------
  [1] D. Delande, These d'Etat: Atomes de Rydberg en champs statiques intenses, Universite Pierre et Marie Curie (Paris),
  1988, https://theses.hal.science/tel-00011864v1
  """
#  if type(alp)==complex:
#    my_type = complex
#  else:
#    my_type = float
  x = np.zeros((nsup-l,nsup-l))
  # n->n
  for n in range(l+1,nsup+1):
    x[n-l-1,n-l-1] = 2.0*n
  # n->n+1
  for n in range(l+1,nsup):
    x[n-l,n-l-1] = math.sqrt((n-l)*(n+l+1))
  # n-> n-1
  for n in range(l+2,nsup+1):
    x[n-l-2,n-l-1] = math.sqrt((n-l-1)*(n+l))
#  print(x)
  return x*alp

def compute_z_matrix(l: int, m: int, lprime: int, nsup: int, alp: complex) -> np.ndarray:
  r"""
  Compute the matrix representing the z operator in a Sturmian basis of parameter alp.

  This is the submatrix connecting the l subspace on the right side and the lprime subspace on the left side.
  It is non-zero only if :math:`|l-lprime| = 1`.
  The dimension of the rectangular matrix is [0:nsup-l,0:nsup-lprime].
  This rectangular sub-matrix matrix is tridiagonal.
  The z operator in the Sturmian basis is, according to Eq. (I-211) of [1], z = alp*(S_3+S_1-T_3-T_1)
  Its matrix elements are given by Eq. (I-197) of [1].

  Parameters
  ----------
  l : int
    Angular momentum quantum number on the left side.
   m: int
    Magnetic quantum number of the two states.
  lprime: int
    Angular momentum quantum number on the right side.
  nsup : int
    Maximal principal quantum number included in the calculation.
  alp : complex or float
    Dilatation parameter

  Returns
  -------
  Numpy array :math:`[0\!:\!nsup-l,0\!:\!nsup-lprime]` of complex or float.
  It should be a tridiagonal matrix.

  References
  ----------
  [1] D. Delande, These d'Etat: Atomes de Rydberg en champs statiques intenses, Universite Pierre et Marie Curie (Paris),
  1988, https://theses.hal.science/tel-00011864v1
  """
#  if type(alp)==complex:
#    my_type = complex
#  else:
#    my_type = float
  x = np.zeros((nsup-l,nsup-lprime))
  if l==lprime+1:
    if abs(m)>lprime:
      return x
# nprime,lprime -> nprime, lprime+1
    for nprime in range(l+1,nsup+1):
      x[nprime-l-1,nprime-lprime-1] = math.sqrt((nprime-lprime-1)*(nprime+lprime+1))*math.sqrt(((lprime+1)**2-m**2)/(4*((lprime+1)**2)-1))
  # nprime, lprime -> nprime+1,lprime+1
    for nprime in range(l,nsup):
      x[nprime-l,nprime-lprime-1] = 0.5*math.sqrt((nprime+lprime+1)*(nprime+lprime+2))*math.sqrt(((lprime+1)**2-m**2)/(4*((lprime+1)**2)-1))
  # nprime, lprime -> nprime-1,lprime+1
    for nprime in range(1+2,nsup+1):
      x[nprime-l-2,nprime-lprime-1] = 0.5*math.sqrt((nprime-lprime-1)*(nprime-lprime-2))*math.sqrt(((lprime+1)**2-m**2)/(4*((lprime+1)**2)-1))
  if l==lprime-1:
    if abs(m)>l:
      return x
  # nprime,lprime -> nprime, lprime-1
    for nprime in range(l+1,nsup+1):
      x[nprime-l-1,nprime-lprime-1] = math.sqrt((nprime-lprime)*(nprime+lprime))*math.sqrt((lprime**2-m**2)/(4*(lprime**2)-1))
  # nprime, lprime -> nprime-1,lprime-1
    for nprime in range(l+2,nsup+1):
      x[nprime-l-2,nprime-lprime-1] = 0.5*math.sqrt((nprime+lprime)*(nprime+lprime-1))*math.sqrt((lprime**2-m**2)/(4*(lprime**2)-1))
  # nprime, lprime -> nprime+1,lprime-1
    for nprime in range(l,nsup):
      x[nprime-l,nprime-lprime-1] = 0.5*math.sqrt((nprime-lprime)*(nprime-lprime+1))*math.sqrt((lprime**2-m**2)/(4*(lprime**2)-1))
#  print(x)
  return x*alp

def compute_2irpz_matrix(l: int, m:int, lprime: int, nsup: int, alp: float) -> np.ndarray:
  """
  Compute the matrix representing the 2*i*r*pz operator in a Sturmian basis of parameter alp.

  This is the submatrix connecting the l subspace on the right side and the lprime subspace on the left side.
  It is non-zero only if :math:`|l-lprime| = 1`.
  The dimension of the rectangular matrix is :math:`[0:nsup-l,0:nsup-lprime]`.
  This rectangular sub-matrix matrix is tridiagonal.
  The r*pz operator in the Sturmian basis is, according to Eq. (I-211) of [1],:math:`r*pz = T_2-S_2`
  Its matrix elements are given by Eq. (I-197) of [1].
  They are purely imaginary, this is why they are multiplied by 2*i to obtain a real result: :math:`2*i*r*pz = T_+-S_++S_--T_-`


  Parameters
  ----------
  l : int
    Angular momentum quantum number on the left side.
  m: int
    Magnetic quantum number of the two states.
  lprime: int
    Angular momentum quantum number on the right side.
  nsup : int
    Maximal principal quantum number included in the calculation.
  alp : float
    Dilatation parameter

  Returns
  -------
  Numpy array [0:nsup-l,0:nsup-lprime] of floats.
  It should be a tridiagonal matrix.

  References
  ----------
  [1] D. Delande, These d'Etat: Atomes de Rydberg en champs statiques intenses, Universite Pierre et Marie Curie (Paris),
  1988, https://theses.hal.science/tel-00011864v1
  """
  x = np.zeros((nsup-l,nsup-lprime))
  if l==lprime+1:
    if abs(m)>lprime:
      return x
  # nprime,lprime -> nprime, lprime+1 is 0
  # nprime, lprime -> nprime+1,lprime+1
    for nprime in range(l,nsup):
      x[nprime-l,nprime-lprime-1] = -math.sqrt((nprime+lprime+1)*(nprime+lprime+2))*math.sqrt(((lprime+1)**2-m**2)/(4*((lprime+1)**2)-1))
  # nprime, lprime -> nprime-1,lprime+1
    for nprime in range(1+2,nsup+1):
      x[nprime-l-2,nprime-lprime-1] = math.sqrt((nprime-lprime-1)*(nprime-lprime-2))*math.sqrt(((lprime+1)**2-m**2)/(4*((lprime+1)**2)-1))
  if l==lprime-1:
    if abs(m)>l:
      return x
  # nprime,lprime -> nprime, lprime-1 is 0
  # nprime, lprime -> nprime-1,lprime-1
    for nprime in range(l+2,nsup+1):
      x[nprime-l-2,nprime-lprime-1] = math.sqrt((nprime+lprime)*(nprime+lprime-1))*math.sqrt((lprime**2-m**2)/(4*(lprime**2)-1))
  # nprime, lprime -> nprime+1,lprime-1
    for nprime in range(l,nsup):
      x[nprime-l,nprime-lprime-1] = -math.sqrt((nprime-lprime)*(nprime-lprime+1))*math.sqrt((lprime**2-m**2)/(4*(lprime**2)-1))
#  print(x)
  return x

def compute_Green_function_matrix(l: int, energy: float, nsup: int, alp: complex) -> np.ndarray:
  """
  Compute the matrix representing the 1/(2r(E-H)) operator in a Sturmian basis of parameter alp.

  One first computes the 2r(E-H) matrix and then invert it
  One has 2r(E-H) = U_3*(-1/alp+2*E*alp) + U_1*(1/alp+2*E*alp) + 2 with U=S+T.
  Its matrix elements are given by Eq. (I-197) of [1].

  Parameters
  ----------
  l : int
    Angular momentum quantum number.
  energy: float
    The ergy at which the Green function is computed.
  nsup : int
    Largest value of the principal quantum number.
  alp : complex or float
    Dilatation parameter

  Returns
  -------
  Numpy array [0:nsup-l,0:nsup-l] of complex or float.
  It should be a tridiagonal matrix.

  References
  ----------
  [1] D. Delande, These d'Etat: Atomes de Rydberg en champs statiques intenses, Universite Pierre et Marie Curie (Paris),
  1988, https://theses.hal.science/tel-00011864v1
  """
  if type(alp)==complex:
    my_type = complex
  else:
    my_type = float
  x = np.zeros((nsup-l,nsup-l),dtype=my_type)
  diagonal_coefficient = -1.0/alp + 2.0*energy*alp
  non_diagonal_coefficient = 1.0/alp + 2.0*energy*alp
  # n->n
  for n in range(l+1,nsup+1):
    x[n-l-1,n-l-1] = diagonal_coefficient*n+2.0
  # n->n+1
  for n in range(l+1,nsup):
    x[n-l,n-l-1] = 0.5*non_diagonal_coefficient*math.sqrt((n-l)*(n+l+1))
  # n->n-1
  for n in range(l+2,nsup+1):
    x[n-l-2,n-l-1] = 0.5*non_diagonal_coefficient*math.sqrt((n-l-1)*(n+l))
  # Invert the matrix
  x = np.linalg.inv(x)
  #print(x)
  return x

def check_orthogonality(n: int, l:int, nprime:int, lprime:int, nsup: int) -> float:
  """
  Compute the overlap between eigenstates of the hydrogen atom.

  Should be unity if n,l = nprime,lprime zero otherwise.
  The calculation is performed in the Sturmian basis of the n,l state, i.e. with scaling parameter alp=n.
  The state nprime,lprime is simple in the alp=nprime Sturmian basis. It is computed in the alp=n Sturmian basis using the
  matrix generated by the routine compute_dilatation_matrix.

  Parameters
  ----------
  n : int
    Principal quantum number of the first state.
  l : int
    Angular momentum of the first state.
  nprime : int
    Principal quantum number of the second state.
  lprime : int
    Angular momentum of the second state.
  nsup : int
    Maximum principal quantum number included in the calculation.

  Returns
  -------
  float
    The overlap < n l|nprime lprime>

  """
# The simplest case
  if l!=lprime:
    return 0.0
  if n<=l or nprime<=lprime:
    return 0.0
# The initial state is the Sturmian basis
# only one component is non-zero (unity)
  initial_state = np.zeros(nsup-l)
  initial_state[nprime-lprime-1] = 1.0
# The gamma dilatation factor necessary to bring the initial state with alp=nprime to alp=n
  gamma = math.log(n/nprime)
# Compute the dilatation matrix
  x = compute_dilatation_matrix(2*l+2, gamma, nsup-l-1)
# Apply it on the initial state
  dilatated_state = x.dot(initial_state)
# compute the 2r matrix in the Sturmian basis with alp=n
  y = compute_2r_matrix(l, nsup, n)
#  print(y.dot(dilatated_state))
# Apply it on the dilatated state and make the overlap with the n,l state
# Do not forget the normalization factor when going from Sturmian state to hydrogenic state : 1/(sqrt(2)*n) for each state
  return y.dot(dilatated_state)[n-l-1]/(2.0*n*nprime)

def compute_dipole_matrix_element(n: int, l:int, m: int, nprime:int, lprime:int) -> float:
  """
  Compute the dipole matrix element between eigenstates of the hydrogen atom.

  Should be zero if :math:`|l-lprime|` is not 1, non-zero otherwise
  The calculation is performed in the Sturmian basis of the n,l state, i.e. with scaling parameter alp=n.
  The state nprime,lprime is simple in the alp=nprime Sturmian basis. It is computed in the alp=n Sturmian basis using the
  matrix generated by the routine compute_dilatation_matrix.
  Then, one applies successively the 2r (for going from Sturmian to hydrogenic scalar product) and the z operator

  Parameters
  ----------
  n : int
    Principal quantum number of the first state.
  l : int
    Angular momentum of the first state.
  m : int
    Magnetic quantum number of the first state.
  nprime : int
    Principal quantum number of the second state.
  lprime : int
    Angular momentum of the second state.

  Returns
  -------
  float
    The matrix element < n l m| z | nprime lprime m>

  """
# The simplest case
  if abs(lprime-l) != 1:
    return 0.0
  if n<=l or nprime<=lprime:
    return 0.0
  nsup = max(200,n+50,nprime+50)
# The initial state is the Sturmian basis
# only one component is non-zero (unity)
  initial_state = np.zeros(nsup-lprime)
  initial_state[nprime-lprime-1] = 1.0
# The gamma dilatation factor necessary to bring the initial state with alp=nprime to alp=n
  gamma = math.log(n/nprime)
# Compute the dilatation matrix
  x = compute_dilatation_matrix(2*lprime+2, gamma, nsup-lprime-1)
# Apply it on the initial state
  dilatated_state = x.dot(initial_state)
# compute the 2r matrix in the Sturmian basis with alp=n
  y = compute_2r_matrix(lprime, nsup, n)
  intermediate_state = y.dot(dilatated_state)
  z = compute_z_matrix(l, m, lprime, nsup, n)
#  print(y.dot(dilatated_state))
# Apply it on the dilatated state and make the overlap with the n,l state
# Do not forget the normalization factor when going from Sturmian state to hydrogenic state : 1/(sqrt(2)*n) for each state
# Finally, as noted in the remark page 82 of [1], the phase convention for Sturmian function is different from the one of hydrogenic states
# The additional phase factor is added here so that one obtains the same result than the usual Gordon formula
  return ((-1)**(n+nprime-l-lprime-2))*z.dot(intermediate_state)[n-l-1]/(2.0*n*nprime)


def compute_dipole_matrix_element_velocity_gauge(n: int, l: int, m:int, nprime:int, lprime:int) -> float:
  """
  Compute the dipole matrix element between eigenstates of the hydrogen atom in the velocity gauge.

  These are the matrix elements of i*p_z (the i makes the result real)
  Should be zero if :math:`|l-lprime|` is not 1, non-zero otherwise
  The calculation is performed in the Sturmian basis of the n,l state, i.e. with scaling parameter alp=n.
  The state nprime,lprime is simple in the alp=nprime Sturmian basis. It is computed in the alp=n Sturmian basis using the
  matrix generated by the routine compute_dilatation_matrix.
  Then, one applies the 2*i*r*pz operator (the 2r is for going from Sturmian to hydrogenic scalar product)

  Parameters
  ----------
  n : int
    Principal quantum number of the first state.
  l : int
    Angular momentum of the first state.
   m : int
    Magnetic quantum number of the first state.
  nprime : int
    Principal quantum number of the second state.
  lprime : int
    Angular momentum of the second state.

  Returns
  -------
  float
    The matrix element < n l m| i*pz | nprime lprime m>

  """
# The simplest case
  if abs(lprime-l) != 1:
    return 0.0
  if n<=l or nprime<=lprime:
    return 0.0
  nsup = max(200,n+50,nprime+50)
# The initial state is the Sturmian basis
# only one component is non-zero (unity)
  initial_state = np.zeros(nsup-lprime)
  initial_state[nprime-lprime-1] = 1.0
# The gamma dilatation factor necessary to bring the initial state with alp=nprime to alp=n
  gamma = math.log(n/nprime)
# Compute the dilatation matrix
  x = compute_dilatation_matrix(2*lprime+2, gamma, nsup-lprime-1)
# Apply it on the initial state
  dilatated_state = x.dot(initial_state)
# compute the 2*i*r*pz matrix in the Sturmian basis with alp=n
  y = compute_2irpz_matrix(l, m, lprime, nsup, n)
#  print(y.dot(dilatated_state))
# Apply it on the dilatated state and make the overlap with the n,l state
# Do not forget the normalization factor when going from Sturmian state to hydrogenic state : 1/(sqrt(2)*n) for each state
# Finally, as noted in the remark page 82 of [1], the phase convention for Sturmian function is different from the one of hydrogenic states
# The additional phase factor is added here so that one obtains the same result than the usual Gordon formula
  return ((-1)**(n+nprime-l-lprime-2))*y.dot(dilatated_state)[n-l-1]/(2.0*n*nprime)

def compute_partial_two_photon_matrix_element(n: int, l:int, m:int, nprime:int, lprime:int, lintermediate: int, nsup: int, alp: float) -> float:
  """
  Compute the two photon matrix element between eigenstates of the hydrogen atom, with intermediate state of well defined angular momentum.

  Should be zero if :math:`|l-lintermediate|` and :math:`|lprime-lintermediate|` are not 1, non-zero otherwise.
  As a consequence, there is the selection rule :math:`|l-lprime|` = 0 or 2.
  The calculation is performed in a Sturmian basis of scaling parameter alp.
  The state nprime,lprime is simple in the alp=nprime Sturmian basis.
  One then applies successively:
    * The z operator in the alp=nprime basis
    * The 2r operator in the alp=nprime basis (for going from Sturmian to hydrogenic scalar product)
    * A dilatation operator to go from scaling parameter nprime to alp
    * The 1/(2r(Eintermediate-H)) operator (Green function in Sturmian basis)
    * Another dilatation operator to go from scaling parameter alp to n
    * The 2r operator in the alp=n basis(to compensate for the Green function)
    * The z operator in the alp=n basis

  Parameters
  ----------
  n : int
    Principal quantum number of the first state.
  l : int
    Angular momentum of the first state.
  m : int
    Magnetic quantum number of the two states.
  nprime : int
    Principal quantum number of the second state.
  lprime : int
    Angular momentum of the second state.
  lintermediate : int
    The angular momentum of the intermediate state.
  nsup : int
    Maximum principal quantum number included in the calculation.
  alp: float
    The scaling parameter of the Sturmian basis used in the calculation

  Returns
  -------
  float
    The matrix element < n l m| z 1/(H-Eintermediate) z | nprime lprime m>

  """
# The simplest case
  if abs(lprime-lintermediate) != 1 or abs(l-lintermediate) != 1:
    return 0.0
  if n<=l or nprime<=lprime:
    return 0.0
  if n==nprime:
    sys.exit('No non-resonant two photon transtion exists when n=nprime!')
# The initial state is the Sturmian basis
# only one component is non-zero (unity)
  initial_state = np.zeros(nsup-lprime)
  initial_state[nprime-lprime-1] = 1.0
# Apply the z operator (only the part ending in the lintermediate angular momentum)
  state = compute_z_matrix(lintermediate, m, lprime, nsup, nprime).dot(initial_state)
# Compute the 2r matrix in the Sturmian basis with alp=nprime and apply it on state
  state = compute_2r_matrix(lintermediate, nsup, nprime).dot(state)
# The gamma dilatation factor necessary to bring the initial state with alp=nprime to alp=alp
  gamma = math.log(alp/nprime)
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
# Energy of the intermediate state
  Eintermediate = -0.25/n**2 - 0.25/nprime**2
# Compute the "Green function" 1/(2*r*(H-Eintermediate)) and apply it on state
  state = compute_Green_function_matrix(lintermediate, Eintermediate, nsup, alp).dot(state)
# The gamma dilatation factor necessary to change from alp=alp to alp=n
  gamma = math.log(n/alp)
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
# Compute the 2r matrix in the Sturmian basis with alp=alp and apply it on state
  state = compute_2r_matrix(lintermediate, nsup, n).dot(state)
# Apply the z operator (only the part ending in the l angular momentum)
  state = compute_z_matrix(l, m, lintermediate, nsup, n).dot(state)
# Make the overlap with the n,l state
# Do not forget the normalization factor when going from Sturmian state to hydrogenic state : 1/(sqrt(2)*n) for each state
# Finally, as noted in the remark page 82 of [1], the phase convention for Sturmian function is different from the one of hydrogenic states
# The additional phase factor is added here
  return ((-1)**(n+nprime-l-lprime-2))*state[n-l-1]/(2.0*n*nprime)

def compute_partial_two_photon_matrix_element_velocity_gauge(n: int, l:int, m: int, nprime:int, lprime:int, lintermediate: int, nsup: int, alp: float) -> float:
  """
  Compute the two photon matrix element between eigenstates of the hydrogen atom, with intermediate state of well defined angular momentum, in the veolcity gauge.

  Should be zero if :math:`|l-lintermediate|` and :math:`|lprime-lintermediate|` are not 1, non-zero otherwise.
  As a consequence, there is the selection rule :math:`|l-lprime|` = 0 or 2.
  The calculation is performed in a Sturmian basis of scaling parameter alp.
  The state nprime,lprime is simple in the alp=nprime Sturmian basis.
  One then applies successively:
    * The 2*i*r*pz operator in the alp=nprime basis
    * A dilatation operator to go from scaling parameter nprime to alp
    * The 1/(2r(Eintermediate)-H) operator (Green function in Sturmian basis)
    * Another dilatation operator to go from scaling parameter alp to n
    * The 2*i*r*pz operator in the alp=n basis


  Parameters
  ----------
  n : int
    Principal quantum number of the first state.
  l : int
    Angular momentum of the first state.
  m : int
    Magnetic quantum number of the two states.
  nprime : int
    Principal quantum number of the second state.
  lprime : int
    Angular momentum of the second state.
  lintermediate : int
    The angular momentum of the intermediate state.
  nsup : int
    Maximum principal quantum number included in the calculation.
  alp: float
    The scaling parameter of the Sturmian basis used in the calculation

  Returns
  -------
  float
    The matrix element < n l m| pz 1/(H-Eintermediate) pz | nprime lprime m>

  """
# The simplest case
  if abs(lprime-lintermediate) != 1 or abs(l-lintermediate) != 1:
    return 0.0
  if n<=l or nprime<=lprime:
    return 0.0
  if n==nprime:
    sys.exit('No non-resonant two photon transtion exists when n=nprime!')
# The initial state is the Sturmian basis
# only one component is non-zero (unity)
  initial_state = np.zeros(nsup-lprime)
  initial_state[nprime-lprime-1] = 1.0
# Apply the 2*i*r*pz operator (only the part ending in the lintermediate angular momentum)
  state = compute_2irpz_matrix(lintermediate, m, lprime, nsup, nprime).dot(initial_state)
# The gamma dilatation factor necessary to bring the initial state with alp=nprime to alp=alp
  gamma = math.log(alp/nprime)
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
# Energy of the intermediate state
  Eintermediate = -0.25/n**2 - 0.25/nprime**2
# Compute the "Green function" 1/(2*r*(H-Eintermediate)) and apply it on state
  state = compute_Green_function_matrix(lintermediate, Eintermediate, nsup, alp).dot(state)
# The gamma dilatation factor necessary to change from alp=alp to alp=n
  gamma = math.log(n/alp)
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
# Apply the z operator (only the part ending in the l angular momentum)
  state = compute_2irpz_matrix(l, m, lintermediate, nsup, n).dot(state)
# Make the overlap with the n,l state
# Do not forget the normalization factor when going from Sturmian state to hydrogenic state : 1/(sqrt(2)*n) for each state
# Finally, as noted in the remark page 82 of [1], the phase convention for Sturmian function is different from the one of hydrogenic states
# The additional phase factor is added here
  return ((-1)**(n+nprime-l-lprime-2))*state[n-l-1]/(2.0*n*nprime)

def compute_full_two_photon_matrix_element(n: int, l: int, m: int, nprime: int, lprime: int, gauge: str='length') -> (float, float):
  """
  Compute the two photon matrix element between eigenstates of the hydrogen atom.

  Should be zero if :math:`|l-lprime|` is not 0 or 2.
  This routine uses compute_partial_two_photon_matrix_element or compute_partial_two_photon_matrix_element_velocity_gauge for each possible intermediate l value.
  It also automatically adjust the alp parameter of the Sturmian basis as well as its size.

  Parameters
  ----------
  n : int
    Principal quantum number of the first state.
  l : int
    Angular momentum of the first state.
  m : int
    Magnetic quantum number of the two states.
  nprime : int
    Principal quantum number of the second state.
  lprime : int
    Angular momentum of the second state.
  gauge: str
    The gauge in which the computation is performed. Either 'length' (default) or 'velocity'

  Returns
  -------
  (float, float)
    The two matrix elements < n l m| z 1/(H-Eintermediate) z | nprime lprime m> involving the two intermediate l values.
    If there is only one possible intermediate l value, the second term is set to 0.

  """
# The simplest case
  if abs(lprime-l) != 2 and abs(lprime-l) != 0:
    return 0.0,0.0
  if n<=l or nprime<=lprime:
    return 0.0,0.0
  if abs(m)>min(l,lprime):
    return 0.0,0.0
# The optimal alp value:
  alp = min(n,nprime)
# This is the maximum size of the basis
# Totally empirical choice |-()
  nsup = 2 * max(n,nprime) + 10
  if gauge=='length':
    compute_partial = compute_partial_two_photon_matrix_element
  else:
    compute_partial = compute_partial_two_photon_matrix_element_velocity_gauge
  if abs(lprime-l)==2:
# Only one intermediate l value is possible
    x1 = compute_partial(n, l, m, nprime, lprime, (l+lprime)//2, nsup, alp)
    x2 = 0.0
  if l==lprime:
    x1 = compute_partial(n, l, m, nprime, lprime, l+1, nsup, alp)
    if l==0:
      x2 = 0.0
    else:
      x2 = compute_partial(n, l, m, nprime, lprime, l-1, nsup, alp)
  if gauge=='velocity':
    omega=0.25/nprime**2-0.25/n**2
    x1 /= omega**2
    x2 /= omega**2
  return x1,x2

def compute_partial_light_shift(n: int, l:int, m: int, lintermediate: int, omega: float, nsup: int, alp: complex) -> complex:
  """
  Compute the two photon matrix element between eigenstates of the hydrogen atom corresponding to the light-shift.

  This computes the partial contribution of states with angular momentum lintermediate
  Should be zero if :math:`|l-lintermediate|` is not 1, non-zero otherwise.
  The calculation is performed in a Sturmian basis of scaling parameter alp.
  The state n,l is simple in the alp=n Sturmian basis.
  One then applies successively:
    * The z operator in the alp=n basis
    * The 2r operator (for going from Sturmian to hydrogenic scalar product)
    * A dilatation operator to go from scaling parameter n to alp
    * The 1/(2r(Eintermediate-H)) operator (Green function in Sturmian basis)
    * Another dilatation operator to go back from scaling parameter alp to n
    * The 2r operator (to compensate for the Green function)
    * The z operator in the alp=n basis

  If one keeps in memory the state after application of z and 2r, the last two steps boil down to a scalar product

  Parameters
  ----------
  n : int
    Principal quantum number of the state.
  l : int
    Angular momentum of the state.
  m: int
    Magnetic quantum number of the state.
  lintermediate : int
    The angular momentum of the intermediate state.
  omega: float
    The frequency of the photon (can be positive or negative).
  nsup : int
    Maximum principal quantum number included in the calculation.
  alp: complex or float
    The scaling parameter of the Sturmian basis used in the calculation

  Returns
  -------
  complex or float
    The matrix element < n l m| z 1/(H-Eintermediate) z | n l m>
  """
# The simplest case
  if abs(l-lintermediate) != 1:
    return 0.0
  if n<=l:
    return 0.0
#  if type(alp)==complex:
#    my_type = complex
#  else:
#    my_type = float
#  print(type(alp))
# The initial state is the Sturmian basis
# only one component is non-zero (unity)
  initial_state = np.zeros(nsup-l)
  initial_state[n-l-1] = 1.0
# Apply the z operator (only the part ending in the lintermediate angular momentum)
  state = compute_z_matrix(lintermediate, m, l, nsup, n).dot(initial_state)
#  print(state)
# Compute the 2r matrix in the Sturmian basis with alp=alp and apply it on state
  state = compute_2r_matrix(lintermediate, nsup, n).dot(state)
# Save state in state1
  state1 = state[:]
# The gamma dilatation factor necessary to bring the initial state with alp=nprime to alp=alp
  if type(alp)==complex:
    gamma = cmath.log(alp/n)
  else:
    gamma = math.log(alp/n)
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
# Energy of the intermediate state
  Eintermediate = -0.5/n**2+omega
# Compute the "Green function" 1/(2*r*(H-Eintermediate)) and apply it on state
  state = compute_Green_function_matrix(lintermediate, Eintermediate, nsup, alp).dot(state)
# The gamma dilatation factor necessary to change from alp=alp to alp=n
  gamma = -gamma
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
# It is now sufficient to make the overlap with state1 and correct normalization
  return np.dot(state,state1)/(2.0*n**2)

def compute_partial_light_shift_velocity_gauge(n: int, l:int, m: int, lintermediate: int, omega: float, nsup: int, alp: complex) -> complex:
  """
  Compute the two photon matrix element between eigenstates of the hydrogen atom corresponding to the light-shift, in the velocity gauge.

  This computes the partial contribution of states with angular momentum lintermediate
  Should be zero if :math:`|l-lintermediate|` is not 1, non-zero otherwise.
  The calculation is performed in a Sturmian basis of scaling parameter alp.
  The state n,l is simple in the alp=n Sturmian basis.
  One then applies successively:
    * The 2*i*r*pzz operator in the alp=n basis
    * A dilatation operator to go from scaling parameter n to alp
    * The 1/(2r(Eintermediate-H)) operator (Green function in Sturmian basis)
    * Another dilatation operator to go back from scaling parameter alp to n
    * The 2*i*r*pz operator in the alp=n basis

  If one keeps in memory the state after application of 2*i*r*pz, the last step boils down to a scalar product

  Parameters
  ----------
  n : int
    Principal quantum number of the state.
  l : int
    Angular momentum of the state.
  m: int
    Magnetic quantum number of the state.
  lintermediate : int
    The angular momentum of the intermediate state.
  omega: float
    The frequency of the photon (can be positive or negative).
  nsup : int
    Maximum principal quantum number included in the calculation.
  alp: complex or float
    The scaling parameter of the Sturmian basis used in the calculation

  Returns
  -------
  complex or float
    The matrix element < n l m| i*pz 1/(H-Eintermediate) i*pz | n l m>
  """
# The simplest case
  if abs(l-lintermediate) != 1:
    return 0.0
  if n<=l:
    return 0.0
#  if type(alp)==complex:
#    my_type = complex
#  else:
#    my_type = float
#  print(type(alp))
# The initial state is the Sturmian basis
# only one component is non-zero (unity)
  initial_state = np.zeros(nsup-l)
  initial_state[n-l-1] = 1.0
# Apply the 2*i*r*pz operator (only the part ending in the lintermediate angular momentum)
  state = compute_2irpz_matrix(lintermediate, m, l, nsup, n).dot(initial_state)
#  print(state)
# Save state in state1
  state1 = state[:]
# The gamma dilatation factor necessary to bring the initial state with alp=nprime to alp=alp
  if type(alp)==complex:
    gamma = cmath.log(alp/n)
  else:
    gamma = math.log(alp/n)
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
# Energy of the intermediate state
  Eintermediate = -0.5/n**2+omega
# Compute the "Green function" 1/(2*r*(H-Eintermediate)) and apply it on state
  state = compute_Green_function_matrix(lintermediate, Eintermediate, nsup, alp).dot(state)
# The gamma dilatation factor necessary to change from alp=alp to alp=n
  gamma = -gamma
# Compute the dilatation matrix and apply it on state
  state = compute_dilatation_matrix(2*lintermediate+2, gamma, nsup-lintermediate-1).dot(state)
#  state = compute_2irpz_matrix(l, lintermediate, nsup, n).dot(state)
#  return state[n-l-1]/(2.0*n**2*omega**2)
# It is now sufficient to make the overlap with state1 and correct normalization
  return np.dot(state,state1)/(2.0*n**2)

def compute_full_light_shift(n: int, l: int, m: int, omega: float, gauge: str='length', debug:bool=False) -> complex:
  """
  Compute the light-shift of a eigenstate of the hydrogen atom.

  This routine uses compute_partial_light_shift or compute_partial_light_shift_velocity_gauge for each possible intermediate l value, and for +/-omega.
  It also automatically adjust the alp parameter of the Sturmian basis as well as its size.

  Parameters
  ----------
  n : int
    Principal quantum number of the state.
  l : int
    Angular momentum of the state.
  m: int
    Magnetic quantum number of the state.
  omega: float
    Frequency of the photon
  gauge: str
    The gauge in which the computation is performed. Either 'length' (default) or 'velocity'
  debug: boolean
    If True, some intermediate values are printed

  Returns
  -------
  complex
    Light-shift (real part) and ionization rate/2 (imaginary part)

  """
# The simplest case
  if n<=l:
    return 0.0
# The optimal alp value:
  alp = 0.5*n-0.3j
# This is the maximum size of the basis
# Totally empirical choice |-()
  nsup = 500
  if gauge=='length':
    compute_partial = compute_partial_light_shift
  else:
    compute_partial = compute_partial_light_shift_velocity_gauge
  x1 = compute_partial(n, l, m, l+1, omega, nsup, alp)
  x2 = compute_partial(n, l, m, l+1, -omega, nsup, alp)
  if debug:
    if gauge=='length':
      print('Contributions of states with l+1 angular momentum for frequencies +omega and -omega and total:',x1,x2,x1+x2)
    else:
      print('Contributions of states with l+1 angular momentum for frequencies +omega and -omega and total:',x1/omega**2,x2/omega**2,(x1+x2)/omega**2)
  if l==0:
    x3 = 0.0
    x4 = 0.0
  else:
    x3 = compute_partial(n, l, m, l-1, omega, nsup, alp)
    x4 = compute_partial(n, l, m, l-1, -omega, nsup, alp)
  if debug:
    if gauge=='length':
      print('Contributions of states with l-1 angular momentum for frequencies +omega and -omega and total:',x3,x4,x3+x4)
    else:
      print('Contributions of states with l-1 angular momentum for frequencies +omega and -omega and total:',x3/omega**2,x4/omega**2,(x3+x4)/omega**2)
  x = x1+x2+x3+x4
  if gauge=='velocity':
    x = (x+1.0)/omega**2
    if debug:
      print('Contributions of the A**2 term in velociy gauge:',1.0/omega**2)
  return x
  return x

def partial_light_shift_3s(omega: float) -> complex:
  """
  Compute the analytic light-shift of the 3s state.

  Based on Eq. (A1) of ref. [1]_.

  WARNING: This result does not agree with the numerical calculation! Could well be wrong.

  Parameters
  ----------
  omega : float
    Frequency of the photon (can be positive or negative)

  Returns
  -------
  complex
    The light shift (real part) and ionization rate/2 (imaginary part)

  References
  ----------
  .. [1] M. haas et al., Phys. Rev. A 73, 052501 (2006)
  """
# Upper energy
  energy = omega-1.0/18.0
  eta = 1.0/cmath.sqrt(-2*energy)
  result = mpmath.hyp2f1(1.0,-eta,1.0-eta,((3-eta)/(3+eta))**2)
  result = 110008287 - 87156324*eta**2 + 29819745*eta**4 - 6022998*eta**6 - 1259712*eta**7 + 2106081*eta**8 + 1912896*eta**9 - 504792*eta**10\
         - 737856*eta**11 - 31041*eta**12 + 84672*eta**13 + 15538*eta**14 + 3456*eta**7*((27-7*eta**2)*2)*(1-eta**2)*result
  result *= -54*eta**2/((9-eta**2)**8)
  return result

def partial_light_shift_1s(omega: float) -> complex:
  """
  Compute the analytic light-shift of the 1s state.

  This uses Eq. (3.11) of Ref. [1]_.

  Parameters
  ----------
  omega : float
    Frequency of the photon (can be positive or negative)

  Returns
  -------
  complex
    The light shift (real part) and ionization rate/2 (imaginary part)

  References
  ----------
  .. [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
# Upper energy
  energy = omega-0.5
  eta = 1.0/cmath.sqrt(-2*energy)
  result = 2*(eta**2)*mpmath.hyp2f1(1.0,-1.0-eta,3.0-eta,((1.0-eta)/(1.0+eta))**2)/(((1.0+eta)**2)*(2.0-eta))
# Alternative expression (should be equal)
#  result = 128*(eta**5)*mpmath.hyp2f1(4.0,2.0-eta,3.0-eta,((1.0-eta)/(1.0+eta))**2)/(((1.0+eta)**8)*(2.0-eta))
# Change sign
  return -result

def partial_light_shift_2s(omega: float) -> complex:
  """
  Compute the analytic light-shift of the 2s state.

  This uses Eq. (3.11) of Ref. [1]_.

  Parameters
  ----------
  omega : float
    Frequency of the photon (can be positive or negative)

  Returns
  -------
  complex
    The light shift (real part) and ionization rate/2 (imaginary part)

  References
  ----------
  .. [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
# Upper energy
  energy = omega-0.125
  eta = 1.0/cmath.sqrt(-2*energy)
  result = 4.0*(eta**5)/((3.0-eta)*((1+0.5*eta)**12))*\
           (4.0*eta**2*mpmath.hyp2f1(6.0,3.0-eta,4.0-eta,((2.0-eta)/(2.0+eta))**2)\
           +8.0*((1.0-0.25*eta**2)**2)*mpmath.hyp2f1(6.0,2.0-eta,5.0-eta,((2.0-eta)/(2.0+eta))**2)/((2.0-eta)*(4.0-eta)))
# Change sign
  return -result

def A_gazeau(n: int, l: int, m: int, l0: int) -> float:
  """
  Compute the A_{ll0} factor of Gazeau, eq. (3.11) of [1]_.

  Parameters
  ----------
  n : int
    principal quantum number
  l : int
    angular momentum
  m : int
    magnetic quantum number
  l0 : int
    another angular momentum

  Returns
  -------
  float
    The A_{ll0} factor

  References
  ----------
  ..  [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  if abs(l-l0)!=1 or l0<0 or n<2:
    return 0.0
  if l==l0+1:
    if abs(m)>=l:
      return 0.0
    else:
      return (n+l)*(n+l-1)*(l**2-m**2)/(n*(n-1)*(2*l+1)*(2*l-1))
  if l==l0-1:
    if abs(m)>=l+1:
      return 0.0
    else:
      return (n-l-1)*(n-l-2)*((l+1)**2-m**2)/(n*(n-1)*(2*l+1)*(2*l+3))

def B_gazeau(n: int, l: int, m: int, l0: int) -> float:
  """
  Compute the B_{ll0} factor of Gazeau, eq. (3.11) of [1]_.

  Parameters
  ----------
  n : int
    principal quantum number
  l : int
    angular momentum
  m : int
    magnetic quantum number
  l0 : int
    another angular momentum

  Returns
  -------
  float
    The B_{ll0} factor

  References
  ----------
  ..  [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  if abs(l-l0)!=1 or l0<0:
    return 0.0
  if l==l0+1:
    if abs(m)>=l:
      return 0.0
    else:
      return (n-l)*(n-l+1)*(l**2-m**2)/(n*(n+1)*(2*l+1)*(2*l-1))
  if l==l0-1:
    if abs(m)>=l+1:
      return 0.0
    else:
     return (n+l+1)*(n+l+2)*((l+1)**2-m**2)/(n*(n+1)*(2*l+1)*(2*l+3))

def C_gazeau(n: int, l: int, m:int, l0: int) -> float:
  """
  Compute the C_{ll0} factor of Gazeau, eq. (3.11) of [1]_.

  Parameters
  ----------
  n : int
    principal quantum number
  l : int
    angular momentum
  m : int
    magnetic quantum number
  l0 : int
    another angular momentum

  Returns
  -------
  float
    The C_{ll0} factor

  References
  ----------
  ..  [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  if abs(l-l0)!=1 or l0<0 or n<2:
    return 0.0
  if l==l0+1:
    if abs(m)>=l:
      return 0.0
    else:
     return np.sqrt((n+l)*(n+l-1)*(n-l)*(n-l+1)/(n**2*(n-1)*(n+1)))*(l**2-m**2)/((2*l+1)*(2*l-1))
  if l==l0-1:
    if abs(m)>=l+1:
      return 0.0
    else:
      return np.sqrt((n-l-1)*(n-l-2)*(n+l+1)*(n+l+2)/(n**2*(n-1)*(n+1)))*((l+1)**2-m**2)/((2*l+1)*(2*l+3))

def S_gazeau(n: int, n0: int, np0: int, l0: int, nu: complex) -> complex:
  """
  Compute the S_{n0,np0}(l0,nu) factor of Gazeau, eq. (3.11) of [1]_.

  Parameters
  ----------
  n : int
    principal quantum number
  n0 : int
    an intermediate principal quantum number
  np0: int
    another intermediate principal quantum number
  l0 : int
    an intemnmediate angular momentum
  nu: complex
    effective principal quantum number of the upper state

  Returns
  -------
  complex
    The S_{n0,np0}(l0,nu) factor

  References
  ----------
  ..  [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)  n : int
  """
  if n0<=l0 or np0<=l0 or l0<0:
    return 0.0
  ninf = min(n0,np0)
  nsup = max(n0,np0)
  r = (1.0-nu/n)/(1.0+nu/n)
  x=0.0
  for q in range(ninf-l0):
    x += (1.0-nu**2/n**2)**(2*q+nsup-ninf)*(4.0*nu/n)**(2*ninf-2*q)*mpmath.hyp2f1(n0+np0,ninf-q-nu,nsup+q+1-nu,r**2)\
        *scipy.special.gamma(ninf-q-nu)*scipy.special.gamma(2*q+nsup-ninf+1)/scipy.special.gamma(q+nsup+1-nu)\
        /(scipy.special.factorial(ninf-l0-1-q)*scipy.special.factorial(ninf+l0-q)*scipy.special.factorial(nsup-ninf+q)*scipy.special.factorial(q))
#    print(n, n0, np0, l0, q, x)
  x *= np.sqrt(n0*np0*scipy.special.factorial(n0-l0-1)*scipy.special.factorial(np0-l0-1)*scipy.special.factorial(n0+l0)*scipy.special.factorial(np0+l0))
  x *= (-1**(n0-np0))*(1.0+nu/n)**(-2*(n0+np0))
  return x

def I_gazeau(n: int, l: int, m: int, omega: float) -> complex:
  """
  Compute the light-shift of the (n,l) state, using eq. (3.11) of [1]_.

  Parameters
  ----------
  n : int
    principal quantum number
  l: int
    angular momentum
  m : int
    magnetic quantum number
  omega : float
    Frequency of the photon (can be positive or negative)

  Returns
  -------
  complex
    The light shift (real part) and ionization rate/2 (imaginary part)

  References
  ----------
  .. [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  energy = omega-0.5/(n**2)
  nu = 1.0/cmath.sqrt(-2*energy)
  """
  print(A_gazeau(n,l,m,l-1))
  print(S_gazeau(n,n-1,n-1,l-1,nu))
  print(B_gazeau(n,l,m,l-1))
  print(S_gazeau(n,n+1,n+1,l-1,nu))
  print(C_gazeau(n, l, m, l-1))
  print(S_gazeau(n, n-1, n+1, l-1, nu))
  print(A_gazeau(n,l,m,l+1))
  print(S_gazeau(n,n-1,n-1,l+1,nu))
  print(B_gazeau(n,l,m,l+1))
  print(S_gazeau(n,n+1,n+1,l+1,nu))
  print(C_gazeau(n, l, m, l+1))
  print(S_gazeau(n, n-1, n+1, l+1, nu))
  """
  x = A_gazeau(n,l,m,l-1)*S_gazeau(n,n-1,n-1,l-1,nu)+B_gazeau(n,l,m,l-1)*S_gazeau(n,n+1,n+1,l-1,nu)-2.0*C_gazeau(n, l, m, l-1)*S_gazeau(n, n-1, n+1, l-1, nu)\
     +A_gazeau(n,l,m,l+1)*S_gazeau(n,n-1,n-1,l+1,nu)+B_gazeau(n,l,m,l+1)*S_gazeau(n,n+1,n+1,l+1,nu)-2.0*C_gazeau(n, l, m, l+1)*S_gazeau(n, n-1, n+1, l+1, nu)
  return 0.25*x*nu/n

def a_gazeau(n: int, l: int) -> float:
  """
  Compute the a(n,l) factor of Gazeau, eq. (D1) of Ref. [1]_.

  Parameters
  ----------
  n : int
    Principal quantum number
  l : int
    Angular momentum

  Returns
  -------
  float
    a(n,l)

  References
  ----------
  .. [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  if n<=1:
    return 0.0
  else:
    return 0.5*math.sqrt((n+l)*(n+l-1)/(n*(n-1)))

def b_gazeau(l: int, m: int) -> float:
  """
  Compute the b(l,m) factor of Gazeau, eq. (D1) of Ref. [1]_.

  Parameters
  ----------
  l : int
    Angular momentum
  m : int
    Magnetic quantum number

  Returns
  -------
  float
    b(l,m)

  References
  ----------
  .. [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  if l==0:
    return 0.0
  else:
    return math.sqrt((l**2-m**2)/((2*l+1)*(2*l-1)))

def c_gazeau(l: int, m: int) -> float:
  """
  Compute the c(l,m) factor of Gazeau, eq. (D1) of Ref. [1]_.

  Parameters
  ----------
  l : int
    Angular momentum
  m : int
    Magnetic quantum number

  Returns
  -------
  float
    c(l,m)

  References
  ----------
  .. [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  if l==0:
    return 0.0
  else:
    return math.sqrt((l+m)*(l+m-1)/((2*l+1)*(2*l-1)))

def T_gazeau(n: int, l: int, m: int, nprime: int, lprime: int) -> float:
  """
  Compute the T factor of Gazeau, eq. (D3) of Ref. [1]_.

  For linear polarization along z, it coincides with C_{N',N} of eq. (D2)

  Parameters
  ----------
  n : int
    Principal quantum number
  l : int
    Angular momentum
  m : int
    Magnetic quantum number
  nprime : int
    Principal quantum number of the other state
  lprime : int
    Angular momentum of the other state

  Returns
  -------
  float
    T_{n l,nprime lprime}^m

  References
  ----------
  .. [1] J.P. Gazeau, J. Math. Phys. 19, 1041 (1978)
  """
  x = 0.0
  if n==nprime+1 and l==lprime+1:
    x += a_gazeau(n,l)*b_gazeau(l,m)
  if n==nprime+1 and l==lprime-1:
    x -= a_gazeau(n,-l-1)*b_gazeau(l+1,m)
  if n==nprime-1 and l==lprime+1:
    x -= a_gazeau(n+1,-l)*b_gazeau(l,m)
  if n==nprime-1 and l==lprime-1:
    x += a_gazeau(n+1,l+1)*b_gazeau(l+1,m)
  return x

def g1(nu: float, n0: int, n: int, nprime: int) -> float:
  """
  Compute the g^1_{0,n')(nu,n0,n) function of the (not numbered) equation, page 238 of Ref. [1]_.

  Parameters
  ----------
  nu : float
    Effective principal quantum number of the intermediate state
  n0 : int
    Principal quantum number of the initial state
  nprime : int
    Another principal quantum number

  Returns
  -------
  float
    The g^1_{0,n')(nu,n0,n) function

  References
  ----------
  .. [1] N.L. Manakov et al., Phys. Lett. A 237, 234 (1998)
  """
  y = (n0-nu)*(n+nu)/((n0+nu)*(n-nu))
  yprime = (n0-nu)*(n-nu)/((n0+nu)*(n+nu))
# print(nu,y,yprime)
#  print(mpmath.appellf1(2-nu,-nprime,nprime+4,3-nu,y,yprime))
  x = nu*(((n-nu)/(n+nu))**nprime)*((16*n0*n*nu**2)/(((n0+nu)**2)*((n+nu)**2)))**2*mpmath.appellf1(2-nu,-nprime,nprime+4,3-nu,y,yprime)/(6*(2-nu))
#  print(x)
  return x

def two_photon_matrix_element_from_1s(n: int, l:int) -> float:
  """
  Compute the two-photon matrix element from 1s state to (n,l) state.

  Of course, one should have l=0 or 2.

  This uses the (not numbered) equations, page 238 of Ref. [1]_.
  Note that the T_{n0,n}(nu) should be multiplied by 0.5 to get the correct results.

  Parameters
  ----------
  n : int
    Principal quantum number of the upper state
  l : int
    Angular momentum of the upper state

  Returns
  -------
  float
    The two-photon matrix element between (n=1,l=0,m=0) and the (n,l,m=0) states

  References
  ----------
  .. [1] N.L. Manakov et al., Phys. Lett. A 237, 234 (1998)
  """
  if l!=0 and l!=2:
    return 0.0
  energy_intermediaire = -0.25-0.25/(n**2)
  nu = 1.0/math.sqrt(-2.0*energy_intermediaire)
  x = math.sqrt(1.0/n)*0.5*((n+1)*(n+2)*g1(nu, 1, n, n-1)-(n-1)*(n-2)*g1(nu, 1, n, n-3))
  omega = 0.25-0.25/(n**2)
  return x/omega**2

def two_photon_matrix_element_from_2s(n: int, l:int) -> float:
  """
  Compute the two-photon matrix element from 2s state to (n,l) state.

  Of course, one should have l=0 or 2.

  This uses the (not numbered) equations, page 238 of Ref. [1]_.
  Note that the T_{n0,n}(nu) should be multiplied by 0.5 to get the correct results.

  Parameters
  ----------
  n : int
    Principal quantum number of the upper state
  l : int
    Angular momentum of the upper state

  Returns
  -------
  float
    The two-photon matrix element between (n=2,l=0,m=0) and the (n,l,m=0) states

  References
  ----------
  .. [1] N.L. Manakov et al., Phys. Lett. A 237, 234 (1998)
  """
  if l!=0 and l!=2:
    return 0.0
  energy_intermediaire = -0.0625-0.25/(n**2)
  nu = 1.0/math.sqrt(-2.0*energy_intermediaire)
  x = math.sqrt(2.0/n)*(0.5*((n+1)*(n+2)*g1(nu, 2, n, n-1)-(n-1)*(n-2)*g1(nu, 2, n, n-3))-64*n**3*nu**2*(((n-2)/(n+2))**n)/(3*(nu**2-4)*((n**2-4)**2)))
  omega = 0.0625-0.25/(n**2)
#  print(nu,omega)
  return x/omega**2
