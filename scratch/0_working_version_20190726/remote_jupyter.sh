On the server:

jupyter notebook --no-browser --port=8889

On your local machineL
ssh -NfL localhost:8888:localhost:8889 -o ProxyCommand='ssh -W %h:%p edwinhs@cartesius.surfsara.nl' edwinhs@fcn1

For the second line, replace the following
- edwinhs with your cartesius user name
- fcn1 with the node number/code that you reserved.

On the browser: 
http://localhost:8888/?token=ac4a0a29133a128b7a2fa408365f3cf6e4e1e9c1f464b847
