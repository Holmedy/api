# api
Run the server :

python app.py
curl -i http://localhost:8080/compute-vector -H "Content-Type: application/json" -H "Authorization:Bearer hubstairs-login" -X POST -d '{"imageUrl" : "my-image-url.com"}'
