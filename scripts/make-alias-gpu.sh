#!/usr/bin/env bash

# get path to root project folder
NEURAL_ENLARGE_PATH="$( cd "$(dirname "$0")" && cd .. ; pwd -P )"

ALIAS_COMMAND="alias enlarge='function ne() { nvidia-docker run --rm -v \"\$(pwd)/\`dirname \${@:\$#}\`\":/ne/input -v \"${NEURAL_ENLARGE_PATH}\":/ne -it jaretburkett/neural-enlarge:gpu \${@:1:\$#-1} \"input/\`basename \${@:\$#}\`\"; }; ne'"

# check if alias exists and replace it if it does
if grep -q "alias enlarge=" ~/.bashrc ;
then
    # alias exists, remove it
    echo -e '\033[0;33mAnother version of the alias exists, replacing it with new version\033[0m'
    if [ "$(uname)" == "Darwin" ];
        then
            # mac has required space for sed i
             sed -i '' '/^alias enlarge=/d' ~/.bashrc
        else
           # assume linux
            sed -i '/^alias enlarge=/d' ~/.bashrc
    fi
fi

# add alias to .bashrc for the user to make it permanent
echo ${ALIAS_COMMAND} >> ~/.bashrc

# run alias command once to make it active for user
eval ${ALIAS_COMMAND}

echo -e "\033[1;32mSuccess! Neural Enlarge alias command created fo user.\033[0m"
echo -e "From now on, you can just run \033[0;36menlarge [options] folder/*.jpg\033[0m from any folder to enlarge files in any folder"
echo -e "\033[0;31mIMPORTANT! keep this project in the same location for the alias to work. If moved, rerun this script from the new location\033[0m"
