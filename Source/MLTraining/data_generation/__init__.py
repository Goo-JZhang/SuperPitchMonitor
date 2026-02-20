#!/usr/bin/env python3
"""
数据生成模块

Usage:
    from data_generation import WindowedSampleGenerator, HDF5DatasetWriter
    
    generator = WindowedSampleGenerator()
    writer = HDF5DatasetWriter("output.hdf5", num_samples=1000)
    writer.create_dataset(generator)
"""

from .timbre_loader import TimbreLoader, get_timbre_loader
from .energy_calculator import Note, FrequencyDomainEnergyCalculator
from .sample_generator import WindowedSampleGenerator
from .hdf5_writer import HDF5DatasetWriter, generate_sanity_check_dataset

__all__ = [
    'TimbreLoader',
    'get_timbre_loader',
    'Note',
    'FrequencyDomainEnergyCalculator',
    'WindowedSampleGenerator',
    'HDF5DatasetWriter',
    'generate_sanity_check_dataset'
]
