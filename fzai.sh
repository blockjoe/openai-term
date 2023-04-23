#!/bin/bash

view-chat(){
  _chat="$(ai list-chats | fzf --preview 'ai view-chat --fzf -c {}')"
  if [ -n "$_chat" ]; then
    _chat=${_chat%%\)*}
    ai view-chat -c "$_chat"
  fi
}

delete-chat(){
  _chats="$(ai list-chats | fzf -m --preview 'ai view-chat --fzf -c {}')"
  for _chat in "${_chats[@]}"; do
    chatname="$(echo "${_chat}" | cut -d ")" -f1)"
    if [ -n "$chatname" ]; then
      ai delete-chat -c "${chatname}"
    fi
  done
}

continue-chat(){
  _chat="$(ai list-chats --fzf | fzf --preview 'ai view-chat --fzf -c {}')"
  if [ -n "$_chat" ]; then
    _chat=${_chat%%\)*}
    if [ "$_chat" == "Begin New Chat" ]; then
      ai new-chat
    else
      ai continue-chat -c "$_chat"
    fi
  fi
}



if ! command -v fzf &> /dev/null; then
	echo "Required dependency fzf not installed.";
elif ! command -v ai &> /dev/null; then
	echo "Required dependency openai-term not installed.";
elif [ -z "$1" ]; then
	continue-chat
else
	case "$1" in
		view)
		  view-chat
		  ;;
		 continue)
		  continue-chat
		  ;;
		delete)
		  delete-chat
		  ;;
	esac
fi
