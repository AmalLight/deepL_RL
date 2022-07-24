#!/bin/bash

# -----------------------------------------------

docker_ps=`docker ps --all`
if [[ "$docker_ps" == *"executor_tmp"* ]] ;
then
     echo "stopping executor_tmp container"
     docker container stop executor_tmp
fi

# -----------------------------------------------

echo "chmod -R 755 on: $common"
      chmod -R 755     $common

echo "docker build"
cd $common/13.Executor/
docker image build -t apache2 .

echo "docker prune"
docker image     prune -f
docker container prune -f
docker volume    prune -f

echo "docker image ls"
      docker image ls

# -----------------------------------------------

function variables () {

    variable=` cat $common/13.Executor/variables.json | jq .$1 `
    variable=${variable//'"'/}
    variable=${variable//','/' '}

    if [[ ${variable::1} == '[' ]] ;
    then
         variable=${variable:1}
         variable=${variable%?}
    fi

    echo "$variable"
}

export reset=`   variables  reset    `
export init=`    variables  init     `

export host=`    variables  host     `
export port=`    variables  port     `

export image=`   variables  image    `
export volume=`  variables  volume   `

export wanted=`  variables  wanted   `
export execute=` variables  execute  `

echo " reset:    $reset    "
echo " init:     $init     "

echo " host:     $host     "
echo " port:     $port     "

echo " image:    $image    "
echo " volume:   $volume   "

echo " wanted:   $wanted   "
echo " execute:  $execute  "

# -----------------------------------------------

volumels=`docker volume ls`
if [[ "$volumels" != *"$volume"* ]] ;
then
     echo "create volume"
     docker volume create $volume
fi

# -----------------------------------------------

if [[ "$reset" == 'yes' ]] ;
then
     echo "reset == yes --> action on volume"
     docker container stop apache2-$port-tmp

     docker volume rm     $volume -f
     docker volume create $volume
fi

echo "running apache2-$port-tmp container"
echo "it will be stopped by the running script and runned again"
export dirv=$volume && bash run.sh

echo "docker volume ls"
      docker volume ls

echo "docker ps --all"
      docker ps --all

# -----------------------------------------------

if [[ "$init" == 'yes' ]] ;
then
     echo "init == yes --> run/init container: $image"

     run="docker run --rm --name executor_tmp -v $volume:/$volume/ -w /$volume/"
     run="$run $image sh -c ' curl http://$host:$port/apacheme.sh -o apacheme.sh ; \
                              bash apacheme.sh /$volume $host $port \"$wanted\" ' "

     echo "run: $run" && eval ${run} && echo 'done.'
fi

if [[ "$execute" == 'yes' ]] ;
then
     echo "execute == yes --> exec action: $software on: $folder/$file"

     cmd_numbers=` variables cmd_numbers `
     echo " cmd numbers : $cmd_numbers "

     base="docker run --rm --name executor_tmp -v $volume:/$volume/ -w /$volume/ -v /dev/shm:/dev/shm"

     if (( $cmd_numbers > 0 )) ;
     then
          for numb in `seq $cmd_numbers`
          do
               echo " numb : $numb "
               numb=$((numb-1))

               numb_folder=` variables "cmd_list[$numb].folder" `
               echo " numb folder : $numb_folder "

               numb_file=` variables "cmd_list[$numb].file" `
               echo " numb file : $numb_file "

               numb_software=` variables "cmd_list[$numb].software" `
               echo " numb software : $numb_software "

               run="$base $image sh -c ' curl http://$host:$port/$numb_folder/$numb_file -o $numb_folder/$numb_file ;"
               run="$run cd $numb_folder ; $numb_software $numb_file '"

               echo "run: $run" && eval ${run}
          done
     fi
fi