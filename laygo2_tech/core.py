# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2014, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################

"""Laygo2 technology setup in Niftylab's style"""
from pathlib import Path

import yaml
from laygo2.object.technology import NiftyTechnology
from .flex import load_flex_templates

LOCAL_CONFIG_KEYS = ("export_template", "import_template", "export")


def _load_yaml_mapping(filename):
    with open(filename, "r") as stream:
        try:
            data = yaml.safe_load(stream) or {}
        except yaml.YAMLError as exc:
            print(exc)
            return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {filename}")
    return data


def _deep_update(base, overlay):
    for key, value in overlay.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_local_config(tech_params, config_params):
    for key in LOCAL_CONFIG_KEYS:
        if key not in config_params:
            continue
        value = config_params[key] or {}
        if isinstance(tech_params.get(key), dict) and isinstance(value, dict):
            _deep_update(tech_params[key], value)
        else:
            tech_params[key] = value
    return tech_params


# Technology parameters
tech_fname = "./laygo2_tech/laygo2_tech.yaml"
config_fname = "./laygo2_config.yaml"
tech_params = _load_yaml_mapping(tech_fname)
if Path(config_fname).exists():
    tech_params = _apply_local_config(tech_params, _load_yaml_mapping(config_fname))
techobj = NiftyTechnology(tech_params = tech_params)

def load_templates_and_grids():
    tlib = techobj.load_tech_templates()
    glib = techobj.load_tech_grids(templates=tlib)
    return tlib, glib 

def load_templates():
    tlib = techobj.load_tech_templates()
    # return tlib  # uncomment if you are not planning to use flexible templates.
    # flexible templates. 
    tlib_flex =load_flex_templates()
    for tn, t in tlib_flex.items():
        tlib.append(t)
    return tlib

def load_grids(templates, libname=None, params=None):
    return techobj.load_tech_grids(templates=templates, libname=libname, params=params)

def generate_cut_layer(dsn,grids,tlib,templates):
    #r23     = grids["routing_23_cmos"]
    #r23_cut = grids["routing_23_cmos_cut"] 
    #dsn.rect_space("M0",r23,r23_cut,150)
    pass

def generate_tap(dsn, grids, tlib, templates, type_iter='nppn', type_extra=None, transform_iter='0X0X', transform_extra=None, side='both'):
    techobj.generate_tap(dsn=dsn, 
                         grids=grids, 
                         tlib=tlib, 
                         templates=templates, 
                         type_iter=type_iter, 
                         type_extra=type_extra, 
                         transform_iter=transform_iter, 
                         transform_extra=transform_extra, 
                         side='both',
                         )

def generate_gbnd(dsn, grids, templates):
    techobj.generate_gbnd(dsn=dsn,
                          grids=grids,
                          templates=templates,
                          )

def generate_pwr_mesh(dsn, grid, track=[None, [0]], pitch=[None, None], 
                      assign_netname=True, netname=[None, ['VSS', 'VDD']], 
                      generate_pin=True, pinname_prefix='', pinname_suffix='', 
                      noindex_to_single_net_pins=True):
    techobj.generate_pwr_mesh(dsn=dsn,
                              grid=grid,
                              track=track.copy(),
                              pitch=pitch.copy(),
                              assign_netname=assign_netname,
                              netname=netname.copy(),
                              generate_pin=generate_pin,
                              pinname_prefix=pinname_prefix,
                              pinname_suffix=pinname_suffix,
                              noindex_to_single_net_pins=noindex_to_single_net_pins,
                              )
                          
def generate_pwr_rail(dsn, grids, tlib=None, templates=None, route_type='cmos', netname=None, vss_name='VSS', vdd_name='VDD', rail_swap=False, vertical=False, pin_num=0, pin_pitch=0):
    techobj.generate_pwr_rail(dsn=dsn, 
                              grids=grids, 
                              tlib=tlib,
                              templates=templates, 
                              route_type=route_type, 
                              netname=netname, 
                              vss_name=vss_name, 
                              vdd_name=vdd_name, 
                              rail_swap=rail_swap, 
                              vertical=vertical, 
                              pin_num=pin_num, 
                              pin_pitch=pin_pitch
                              )

def extend_wire(dsn, layer='M4', target=500):
    techobj.extend_wire(dsn=dsn, layer=layer, target=target)

def fill_by_instance(dsn, grids, tlib, templates, inst_name, canvas_area="full", shape=[1,1], iter_type=("R0","MX"), pattern_direction='v', fill_sort='filler'):
    if isinstance(inst_name, tuple):
        inst_name = list(inst_name)
    if isinstance(iter_type, tuple):
        iter_type = list(iter_type)
    techobj.fill_by_instance(dsn=dsn, 
                             grids=grids, 
                             tlib=tlib, 
                             templates=templates, 
                             inst_name=inst_name,
                             canvas_area=canvas_area,
                             shape=shape.copy(),
                             iter_type=iter_type.copy(),
                             pattern_direction=pattern_direction,
                             fill_sort=fill_sort,
                             )

def post_process( dsn, grids, tlib, templates ):
    techobj.post_process(dsn=dsn, grids=grids, tlib=tlib, templates=templates)


