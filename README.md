# hame
Compute one-photon and two-photon matrix elements between eigenstates of the hydrogen atom

This library computes one-photon and two-photon matrix elements between hydrogenic states.
It includes diagonal elements of the two-photon matrix elements, that is light-shift and one-photon ionization rate.

It performs the numerical calculation in the length gauge in a Sturmian basis.
It additionally performs the calculation in the velocity gauge.

It checks that results in length and velocity gauges are equal.

A simple script using the library is compute_two_photon_matrix_element.py
