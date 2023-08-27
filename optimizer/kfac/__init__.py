"""Top-level module for K-FAC."""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import sys

import optimizer.kfac.assignment as assignment
import optimizer.kfac.base_preconditioner as base_preconditioner
import optimizer.kfac.distributed as distributed
import optimizer.kfac.enums as enums
import optimizer.kfac.gpt_neox as gpt_neox
import optimizer.kfac.layers as layers
import optimizer.kfac.preconditioner as preconditioner
import optimizer.kfac.scheduler as scheduler
import optimizer.kfac.tracing as tracing
import optimizer.kfac.warnings as warnings
