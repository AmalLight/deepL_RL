#!/bin/bash

PWD_save=$PWD

autosudo

cd /home/kaumi/Git/Anti_Hackers

if (( ${#@} < 2 )) || [[ $2 != 'no'  ]] ; then                  apache2 8080     ; fi
if (( ${#@} > 1 )) && [[ $2 == 'no'  ]] ; then sudo docker stop apache2-8080-tmp ; fi

if (( ${#@} < 1 )) || [[ $1 != 'yes' ]] ; then sudo docker stop webmin-10000-tmp ; fi
if (( ${#@} > 0 )) && [[ $1 == 'yes' ]] ; then                  webmin 10000     ; fi

cd /home/kaumi/Git/deepL_RL

if (( ${#@} < 2 )) || [[ $2 != 'no'  ]] ; then                  apache2 8081     ; fi
if (( ${#@} > 1 )) && [[ $2 == 'no'  ]] ; then sudo docker stop apache2-8081-tmp ; fi

if (( ${#@} < 1 )) || [[ $1 != 'yes' ]] ; then sudo docker stop webmin-10001-tmp ; fi
if (( ${#@} > 0 )) && [[ $1 == 'yes' ]] ; then                  webmin 10001     ; fi

cd $PWD_save

echo ''
sudo docker ps --all
echo ''
