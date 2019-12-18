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
