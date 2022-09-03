#! /usr/bin/env bash

# license server
source /tools/flexlm/flexlm.sh

# Cadence settings
export SPECTRE_DEFAULTS=-E
export CDS_Netlisting_Mode="Analog"
export CDS_AUTO_64BIT=ALL

# setup virtuoso
export CDS_INST_DIR=/tools/cadence/IC/IC617
export SPECTRE=/tools/cadence/SPECTRE/SPECTRE191
export MMSIM_HOME=/tools/cadence/MMSIM151
export CDSHOME=$CDS_INST_DIR

# PATH setup
export PATH="${CDS_INST_DIR}/tools/plot/bin:${PATH}"
export PATH="${CDS_INST_DIR}/tools/dfII/bin:${PATH}"
export PATH="${CDS_INST_DIR}/tools/bin:${PATH}"
export PATH="${SPECTRE}/bin:${PATH}"
export PATH="${MMSIM_HOME}/tools/bin:${PATH}"

### Setup BAG
source .bashrc_bag
