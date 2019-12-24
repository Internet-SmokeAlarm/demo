# fedlearn-demo
Demo for Federated Learning System

## Client

### Running the client

```
python3 client.py 5560
```

### Testing the client

```
curl -X POST -H "Content-Type: application/json" -d '{"device_id" : 234243, "device_api_key" : 123000}' http://localhost:5560/register
```

## Server

### Running the server

```
python3 server.py
```

## Common Errors

If running Mac OS X there are security issues with allowing a python process to call fork(). See the following for information how to resolve this error: ```https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr/52230415```
