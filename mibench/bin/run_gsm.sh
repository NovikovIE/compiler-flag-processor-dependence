#!/bin/bash
./toast_${PLATFORM} -fps -c ../input_data/gsm_input.au > gsm_output.encode.gsm
./untoast_${PLATFORM} -fps -c gsm_output.encode.gsm > gsm_output.decode.gsm
