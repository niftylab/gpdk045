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
import yaml
from laygo2.object.technology import NiftyTechnology

# Technology parameters
tech_fname = './laygo2_tech/laygo2_tech.yaml'
with open(tech_fname, 'r') as stream:
    try:
        tech_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
techobj = NiftyTechnology(tech_params = tech_params)

def load_templates():
    return techobj.load_tech_templates() 

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

def fill_by_instance(dsn, grids, tlib, templates, inst_name:tuple, canvas_area="full", shape=[1,1], iter_type=("R0","MX"), pattern_direction='v', fill_sort='filler'):
    techobj.fill_by_instance(dsn=dsn, 
                             grids=grids, 
                             tlib=tlib, 
                             templates=templates, 
                             inst_name=inst_name,
                             canvas_area=canvas_area,
                             shape=shape,
                             iter_type=iter_type,
                             pattern_direction=pattern_direction,
                             fill_sort=fill_sort,
                             )

def post_process( dsn, grids, tlib, templates ):
    techobj.post_process(dsn=dsn, grids=grids, tlib=tlib, templates=templates)


