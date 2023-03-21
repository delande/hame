#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2023 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.1"
#
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
# compute_two_photon_matrix_element.py
# Author: Dominique Delande
# Release date: February, 24, 2023

"""
Created on Tue Feb 21 17:29:59 2023

@author: delande

This program computes one-photon and two-photon matrix elements between hydrogenic states.
It includes diagonal elements of the two-photon matrix elements, that is light-shift and one-photon ionization rate.

It performs the numerical calculation in the length gauge in a Sturmian basis.
It additionally performs the calculation in the velocity gauge.

It checks that results in length and velocity gauges are equal.
"""

#import numpy as np
#import math
#import cmath
#import scipy.special
#import sys
import hame



def main():
  """
  Compute the one-photon and two-photon matrix elements (including light-shoft) between hydrogenic states.

  Returns
  -------
  None.
  """
  """
  n = 2
  l = 1
  nprime = 2
  lprime = 0
  print('Example of one-photon transition')
  print('<',n,l,'| z |',nprime,lprime,'> from Gordon formula    :',hame.gordon_formula(n, l, nprime, lprime))
#  print(check_orthogonality(n, l, nprime, lprime, nsup))
  print('<',n,l,'| z |',nprime,lprime,'> from numerics          :',hame.compute_dipole_matrix_element(n, l, nprime, lprime))
  result_pz = hame.compute_dipole_matrix_element_velocity_gauge(n, l, nprime, lprime)
  print('<',n,l,'| i*pz |',nprime,lprime,'> from numerics       :',result_pz)
  if n!=nprime:
    print('<',n,l,'| i*pz/omega |',nprime,lprime,'> from numerics :',result_pz/(0.5/n**2-0.5/nprime**2))
  print()

  gamma = np.log(n/nprime)
  nsup = 200
  matrix = hame.compute_dilatation_matrix(2*l, gamma, nsup)
  my_result = -n*nprime*np.sqrt((l**2)/(4*l**2-1))/(n**2-nprime**2)
  my_result *= (np.sqrt((n-l)*(n-l+1))*matrix[n-l+1,nprime-lprime-1]-np.sqrt((n+l)*(n+l-1))*matrix[n-l-1,nprime-lprime-1])
  print('my result = ', my_result)
  print()
  """
  """
  n = 5
  l = 1
  nprime = 5
  lprime = 0
# Diagonalize U_1 in the eigenbasis of U_3
  nsup = 20
  my_matrix = np.zeros((nsup,nsup))
  for i in range(nsup-1):
    my_matrix[i+1,i] = 0.5*math.sqrt((i+1)*(i+2*l+2))
    my_matrix[i,i+1] = my_matrix[i+1,i]
  w, v = np.linalg.eigh(my_matrix)
#  print(w)
# select the eigenvalue closest to -n
  my_eigenvalue = np.argmin(abs(w+n))
  print(my_eigenvalue,w[my_eigenvalue])
  print(v[:,my_eigenvalue])
  gamma = math.log(w[-my_eigenvalue]/nprime)+0.5j*np.pi
  gamma = math.log(n/nprime)+0.5j*np.pi
  matrix = hame.compute_dilatation_matrix(2*l+2, gamma, nsup)
  print(matrix[:,nprime-l-1])
  """

  """
  gamma = math.log(n/nprime)-0.5j*np.pi
  print(type(gamma))
  nsup = 200
  matrix = hame.compute_dilatation_matrix(2*l, gamma, nsup)
  my_result = -n*nprime*np.sqrt((l**2)/(4*l**2-1))/(n**2+nprime**2)
  my_result *= (np.sqrt((n-l)*(n-l+1))*matrix[n-l+1,nprime-lprime-1]-np.sqrt((n+l)*(n+l-1))*matrix[n-l-1,nprime-lprime-1])
  my_result = 2.0*np.pi*my_result**2
  print('my result = ', my_result)
  print()
  """

  """
  n = 1
  l = 0
  nprime = 4
  lprime = 2
  print('Example of two-photon transition')
  print('Transition n =',nprime,'l =',lprime,'to n =',n,'l =',l)
  x1,x2 = hame.compute_full_two_photon_matrix_element(n, l, nprime, lprime, 'length')
  if abs(l-lprime)==2:
    print('Contribution of l =',(l+lprime)//2,'in length   gauge:',x1)
  if l==lprime:
    print('Contribution of l =',l+1,'in length   gauge:',x1)
    if l!=0:
      print('Contribution of l =',l-1,'in length   gauge:',x2)
  print('Total matrix element  in length   gauge:', x1+x2)
  x1,x2 = hame.compute_full_two_photon_matrix_element(n, l, nprime, lprime, 'velocity')
  if abs(l-lprime)==2:
    print('Contribution of l =',(l+lprime)//2,'in velocity gauge:',x1)
  if l==lprime:
    print('Contribution of l =',l+1,'in velocity gauge:',x1)
    if l!=0:
      print('Contribution of l =',l-1,'in velocity gauge:',x2)
  print('Total matrix element  in velocity gauge:', x1+x2)
  print()



  n = 2
  l = 0
  nprime = 100
  lprime = 0
  omega = 0.25/n**2 - 0.25/nprime**2
  print('Example of light-shift on a two-photon transition')
  x = hame.compute_full_light_shift(n, l, omega, gauge='length')
  print('Light-shift     of the n =',n,'l =',l,'state at |omega| =',abs(omega),':',x.real,' (length gauge)')
  print('Ionization rate of the n =',n,'l =',l,'state at |omega| =',abs(omega),':',x.imag*2.0,' (length gauge)')
  x = hame.compute_full_light_shift(n, l, omega, gauge='velocity')
  print('Light-shift     of the n =',n,'l =',l,'state at |omega| =',abs(omega),':',x.real,' (velocity gauge)')
  print('Ionization rate of the n =',n,'l =',l,'state at |omega| =',abs(omega),':',x.imag*2.0,' (velocity gauge)')
  print()
  x = hame.compute_full_light_shift(nprime, lprime, omega, gauge='length')
  print('Light-shift     of the n =',nprime,'l =',lprime,'state at |omega| =',abs(omega),':',x.real,' (length gauge)')
  print('Ionization rate of the n =',nprime,'l =',lprime,'state at |omega| =',abs(omega),':',x.imag*2.0,' (length gauge)')
  x = hame.compute_full_light_shift(nprime, lprime, omega, gauge='velocity')
  print('Light-shift     of the n =',nprime,'l =',lprime,'state at |omega| =',abs(omega),':',x.real,' (velocity gauge)')
  print('Ionization rate of the n =',nprime,'l =',lprime,'state at |omega| =',abs(omega),':',x.imag*2.0,' (velocity gauge)')
  """

  omega = 0.2
  n = 2
  l = 0
#  print(hame.ionization_rate_1s(omega))
  x = hame.compute_full_light_shift(n, l, omega, gauge='length', debug=False)
  print('Light-shift     of the n =',n,'l =',l,'state at |omega| =',abs(omega),':',x.real,' (length gauge)')
  print('Ionization rate of the n =',n,'l =',l,'state at |omega| =',abs(omega),':',x.imag*2.0,' (length gauge)')
  y = hame.ionization_rate(n, l, l+1, omega, debug=False)
  if (l>0):
    y += hame.ionization_rate(n, l, l-1, omega, debug=False)
  print('Ionization rate computed using the Fermi golden rule :',y,' (analytic result)')

if __name__ == "__main__":
  main()


  """
  Old stuff

  n = 20
  l = 2
  nprime = 8
  lprime = 2
  lintermediate = 3
  nsup = 30
  for i in range(11):
    alp = 5.0 + 1.0*i
    print(alp,compute_partial_two_photon_matrix_element(n, l, nprime, lprime, lintermediate, nsup, alp))
  return
  """
  """
  n = 2
  l = 0
  nprime = 20
  lprime = 0
  lintermediate = 1
  omega = 0.25/n**2 - 0.25/nprime**2
#  omega = 0.0625
  nsup = 10
  for i in range(11):
    alp = 1.0+0.5*i+0.2j
    print(alp,compute_partial_light_shift_velocity_gauge(n, l, lintermediate, -omega, nsup, alp))
  return
  """

  """
  lintermediate = 0
  alp = 3.0+0.2j
  nsup = 200
  print(1.0/omega**2)
  xplus  = compute_partial_light_shift(n, l, lintermediate, omega, nsup, alp)
  xminus = compute_partial_light_shift(n, l, lintermediate, -omega, nsup, alp)
  print(xplus+xminus, xplus, xminus)
  xplus = compute_partial_light_shift_velocity_gauge(n, l, lintermediate, omega, nsup, alp)
  xminus = compute_partial_light_shift_velocity_gauge(n, l, lintermediate, -omega, nsup, alp)
  print((xplus+xminus)/omega**2, xplus/omega**2, xminus/omega**2)
  """
