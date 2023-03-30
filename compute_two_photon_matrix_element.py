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
import math
#import cmath
import scipy.special
#import sys
import hame



def main():
  """
  Compute the one-photon and two-photon matrix elements (including light-shoft) between hydrogenic states.

  Returns
  -------
  None.
  """

  want_numerical_results = False
  want_analytic_results = True
  want_SI_results_for_beta = True
  conversion_factor_from_atomic_to_SI_results = 4.687125e-6

  """
  n = 8
  l = 3
  nprime = 3
  lprime = 2
  m = 1
  print('One-photon transition')
  if want_analytic_results:
    print('<',n,l,m,'| z |',nprime,lprime,m,'> from Gordon formula    :',hame.gordon_formula(n, l, m, nprime, lprime))
#  print(check_orthogonality(n, l, nprime, lprime, nsup))
  if want_numerical_results:
    print('<',n,l,m,'| z |',nprime,lprime,m,'> from numerics          :',hame.compute_dipole_matrix_element(n, l, m, nprime, lprime))
    result_pz = hame.compute_dipole_matrix_element_velocity_gauge(n, l, m, nprime, lprime)
    print('<',n,l,m,'| i*pz |',nprime,lprime,m,'> from numerics       :',result_pz)
    if n!=nprime:
      print('<',n,l,m,'| i*pz/omega |',nprime,lprime,m,'> from numerics :',result_pz/(0.5/n**2-0.5/nprime**2))
  print()
  """



  n = 2
  l = 0
  nprime = 8
  lprime = 0
  m = 0
  print('Two-photon transition')
  print('Transition n =',n,'l =',l,'m = ', m,'to n =',nprime,'l =',lprime,'m = ',m)
  if want_numerical_results:
    x1,x2 = hame.compute_full_two_photon_matrix_element(n, l, m, nprime, lprime, 'length')
    if abs(l-lprime)==2:
      print('Contribution of l =',(l+lprime)//2,'in length   gauge:',x1)
    if l==lprime:
      print('Contribution of l =',l+1,'in length   gauge:',x1)
      if l!=0:
        print('Contribution of l =',l-1,'in length   gauge:',x2)
    print('Total matrix element  in length   gauge:', x1+x2)
    x1,x2 = hame.compute_full_two_photon_matrix_element(n, l, m, nprime, lprime, 'velocity')
    if abs(l-lprime)==2:
      print('Contribution of l =',(l+lprime)//2,'in velocity gauge:',x1)
    if l==lprime:
      print('Contribution of l =',l+1,'in velocity gauge:',x1)
      if l!=0:
        print('Contribution of l =',l-1,'in velocity gauge:',x2)
    print('Total matrix element  in velocity gauge:', x1+x2)

  if want_analytic_results:
    if n==1 and l==0 and lprime==0:
      print('Matrix element from Manakov et al.:',hame.two_photon_matrix_element_from_1s_to_ns_Manakov(nprime))
    if n==2 and l==0 and lprime==0:
      print('Matrix element from Manakov et al.:',hame.two_photon_matrix_element_from_2s_to_ns_Manakov(nprime))
    if n==1 and l==0:
      print('Matrix element from Marian 1s     :',hame.two_photon_matrix_element_from_1s_Marian(nprime,lprime))
    print('Matrix element from Gazeau        :',hame.two_photon_matrix_element_Gazeau(n,l,m,nprime,lprime),'    WARNING, can be wrong!')
    x = hame.two_photon_matrix_element_Marian(n,l,m,nprime,lprime)
    print('Matrix element from Marian        :',x)
    if max(l,lprime)==2 and min(l,lprime)==0 and m==0:
      beta2 = x*math.sqrt(5)
      if want_SI_results_for_beta:
        beta2 *= conversion_factor_from_atomic_to_SI_results
      print('beta^(2) =',beta2)
    if l==0 and lprime==0 and m==0:
      beta0 = x
      if want_SI_results_for_beta:
        beta0 *= conversion_factor_from_atomic_to_SI_results
      print('beta^(0) =',beta0)
  print()




  n = 2
  l = 0
  nprime = 8
  lprime = 2
  m = 0
  if abs(m)>min(l,lprime) or n<=l or nprime<=lprime:
    print('At least one of the states n =',n,'l =',l,'m =',m,' or n =',nprime,'l =',lprime,'m =',m,' does not exist!')
    return
  omega = 0.25/n**2 - 0.25/nprime**2
  print('Light-shift on a two-photon transition from n =',n,'l =',l,'m =', m,'to n =',nprime,'l =',lprime,'m =',m)
  print()
  if want_numerical_results:
    x = hame.compute_full_light_shift(n, l, m, omega, gauge='length')
    print('Light-shift     of the n =',n,'l =',l,'m =',m,'state at |omega| =',abs(omega),':',x.real,' (length gauge)')
    print('Ionization rate of the n =',n,'l =',l,'m =',m,'state at |omega| =',abs(omega),':',x.imag*2.0,' (length gauge)')
    x = hame.compute_full_light_shift(n, l, m, omega, gauge='velocity')
    print('Light-shift     of the n =',n,'l =',l,'m =',m,'state at |omega| =',abs(omega),':',x.real,' (velocity gauge)')
    print('Ionization rate of the n =',n,'l =',l,'m =',m,'state at |omega| =',abs(omega),':',x.imag*2.0,' (velocity gauge)')
    print()
    x = hame.compute_full_light_shift(nprime, lprime, m, omega, gauge='length')
    print('Light-shift     of the n =',nprime,'l =',lprime,'m =',m,'state at |omega| =',abs(omega),':',x.real,' (length gauge)')
    print('Ionization rate of the n =',nprime,'l =',lprime,'m =',m,'state at |omega| =',abs(omega),':',x.imag*2.0,' (length gauge)')
    x = hame.compute_full_light_shift(nprime, lprime, m, omega, gauge='velocity')
    print('Light-shift     of the n =',nprime,'l =',lprime,'m =',m,'state at |omega| =',abs(omega),':',x.real,' (velocity gauge)')
    print('Ionization rate of the n =',nprime,'l =',lprime,'m =',m,'state at |omega| =',abs(omega),':',x.imag*2.0,' (velocity gauge)')
    print()
  if want_analytic_results:
    zg0 = (hame.I_gazeau(n,l,0,omega)+hame.I_gazeau(n,l,0,-omega)+1.0)/omega**2
    print('Light-shift     of the n =',n,'l =',l,'m =',m,'state computed using the Gazeau formula :',zg0.real,' (analytic result)')
    print('Ionization rate of the n =',n,'l =',l,'m =',m,'state computed using the Gazeau formula :',2.0*zg0.imag,' (analytic result)')
# This is for an initial S state, the beta coefficient is trivial
    if l==0:
      beta0 = zg0
      if want_SI_results_for_beta:
        beta0 *= conversion_factor_from_atomic_to_SI_results
      print('beta^(0)_ac =',beta0.real,'  beta^(0)_ion =',2*beta0.imag)
      print('The light-shift and ionization rate of the J=1/2 m_J=+/-1/2 states are identical to the one of the m=0 state printed above')
      print()
# This is for a D state
    if l==2 and m==0:
# One needs an additional calculation to extract both beta0 and beta2
# This is done using the m=2 state
      zg2 = (hame.I_gazeau(n,l,2,omega)+hame.I_gazeau(n,l,2,-omega)+1.0)/omega**2
      beta0 = math.sqrt(5)*0.5*(zg0+zg2)
      beta2 = math.sqrt(70)*0.25*(zg2-zg0)
      light_shift_fine_structure_three_half = beta0/math.sqrt(5)-beta2*math.sqrt(70)/50
      light_shift_fine_structure_five_half = beta0/math.sqrt(5)-beta2*8/(5*math.sqrt(70))
      if want_SI_results_for_beta:
        beta0 *= conversion_factor_from_atomic_to_SI_results
        beta2 *= conversion_factor_from_atomic_to_SI_results
      print('beta^(0)_ac =',beta0.real,'  beta^(0)_ion =',2*beta0.imag)
      print('beta^(2)_ac =',beta2.real,'  beta^(2)_ion =',2*beta2.imag)
      print('Light-shift     of the n =',n,'l = 2 j = 3/2 mj = +/-1/2 states',light_shift_fine_structure_three_half.real)
      print('Ionization rate of the n =',n,'l = 2 j = 3/2 mj = +/-1/2 states',2*light_shift_fine_structure_three_half.imag)
      print('Light-shift     of the n =',n,'l = 2 j = 5/2 mj = +/-1/2 states',light_shift_fine_structure_five_half.real)
      print('Ionization rate of the n =',n,'l = 2 j = 5/2 mj = +/-1/2 states',2*light_shift_fine_structure_five_half.imag)
      print()
    ze0 = (hame.I_gazeau(nprime,lprime,0,omega)+hame.I_gazeau(nprime,lprime,0,-omega)+1.0)/omega**2
    print('Light-shift     of the n =',nprime,'l =',lprime,'m =',m,'state computed using the Gazeau formula :',ze0.real,' (analytic result)')
    print('Ionization rate of the n =',nprime,'l =',lprime,'m =',m,'state computed using the Gazeau formula :',2.0*ze0.imag,' (analytic result)')
# This is for a final S state, the beta coefficient is trivial
    if lprime==0:
      beta0 = ze0
      if want_SI_results_for_beta:
        beta0 *= conversion_factor_from_atomic_to_SI_results
      print('beta^(0)_ac =',beta0.real,'  beta^(0)_ion =',2*beta0.imag)
      print('The light-shift and ionization rate of the J=1/2 m_J=+/-1/2 states are identical to the one of the m=0 state printed above')
      print()
# This is for a D state
    if lprime==2 and m==0:
# One needs an additional calculation to extract both beta0 and beta2
# This is done using the m=2 state
      ze2 = (hame.I_gazeau(nprime,lprime,2,omega)+hame.I_gazeau(nprime,lprime,2,-omega)+1.0)/omega**2
      beta0 = math.sqrt(5)*0.5*(ze0+ze2)
      beta2 = math.sqrt(70)*0.25*(ze2-ze0)
      light_shift_fine_structure_three_half = beta0/math.sqrt(5)-beta2*math.sqrt(70)/50
      light_shift_fine_structure_five_half = beta0/math.sqrt(5)-beta2*8/(5*math.sqrt(70))
      if want_SI_results_for_beta:
        beta0 *= conversion_factor_from_atomic_to_SI_results
        beta2 *= conversion_factor_from_atomic_to_SI_results
      print('beta^(0)_ac =',beta0.real,'  beta^(0)_ion =',2*beta0.imag)
      print('beta^(2)_ac =',beta2.real,'  beta^(2)_ion =',2*beta2.imag)
      print('Light-shift     of the n =',nprime,'l = 2 j = 3/2 mj = +/-1/2 states',light_shift_fine_structure_three_half.real)
      print('Ionization rate of the n =',nprime,'l = 2 j = 3/2 mj = +/-1/2 states',2*light_shift_fine_structure_three_half.imag)
      print('Light-shift     of the n =',nprime,'l = 2 j = 5/2 mj = +/-1/2 states',light_shift_fine_structure_five_half.real)
      print('Ionization rate of the n =',nprime,'l = 2 j = 5/2 mj = +/-1/2 states',2*light_shift_fine_structure_five_half.imag)
      print()

if __name__ == "__main__":
  main()


  """
  Old stuff
  """


  """
# This is useful if a light-shift is needed, for an arbitrary frequency
# not necessarily on a two-photon transition
  omega = 0.05859375
  n = 8
  l = 1
  m = 1
  if abs(m)>l or n<=l:
    print('The n =',n,'l =',l,'m =',m,'state does not exist!')
    return
#  print(hame.ionization_rate_1s(omega))
  x = hame.compute_full_light_shift(n, l, m, omega, gauge='length', debug=False)
  print('Light-shift     of the n =',n,'l =',l,'m =',m,'state at |omega| =',abs(omega),':',x.real,' (length gauge)')
  print('Ionization rate of the n =',n,'l =',l,'m =',m,'state at |omega| =',abs(omega),':',x.imag*2.0,' (length gauge)')
  y = hame.ionization_rate(n, l, m, l+1, omega, debug=False)
  if (l>0):
    y += hame.ionization_rate(n, l, m, l-1, omega, debug=False)
  print('Ionization rate computed using the Fermi golden rule :',y,' (analytic result)')
# This is specfic to the 2s state
#  z = (hame.partial_light_shift_2s(omega)+hame.partial_light_shift_2s(-omega)+1.0)/omega**2
# This is specfic to the 1s state
#  z = (hame.partial_light_shift_1s(omega)+hame.partial_light_shift_1s(-omega)+1.0)/omega**2
# This is the generic case using the general Gazeau formula
# Add the +/- contributions and the 1/omega**2 contribution from the A**2 term in velocity gauge
  z = (hame.I_gazeau(n,l,m,omega)+hame.I_gazeau(n,l,m,-omega)+1.0)/omega**2
  print('Light-shift computed using the Gazeau formula        :',z.real,' (analytic result)')
  print('Ionization rate computed using the Gazeau formula    :',2.0*z.imag,' (analytic result)')
  return
  """

  """
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
  gamma = np.log(n/nprime)
  nsup = 200
  matrix = hame.compute_dilatation_matrix(2*l, gamma, nsup)
  my_result = -n*nprime*np.sqrt((l**2)/(4*l**2-1))/(n**2-nprime**2)
  my_result *= (np.sqrt((n-l)*(n-l+1))*matrix[n-l+1,nprime-lprime-1]-np.sqrt((n+l)*(n+l-1))*matrix[n-l-1,nprime-lprime-1])
  print('my result = ', my_result)
  print()
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
  energy_intermediaire = -0.25/(nprime**2)-0.25/(n**2)
  omega = 0.25/(nprime**2)-0.25/(n**2)
  tau = 1.0/math.sqrt(-2.0*energy_intermediaire)
  print(hame.b_Marian(n, l, nprime, lprime, 1, 1, tau))
  r1 = (1-tau)/(1+tau)
  r2 = (2-tau)/(2+tau)
  x = 2**(9.5)*3*tau**5*(scipy.special.hyp2f1(5,2-tau,3-tau,r1*r2)-(2-tau)*r1*scipy.special.hyp2f1(5,3-tau,4-tau,r1*r2)/(r2*(3-tau)))/((1+tau)**4*(2+tau)**5)
  print(x/(3*omega**2))
  """
