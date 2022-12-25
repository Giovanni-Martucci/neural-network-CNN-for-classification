const express = require('express')
const app = express()
const port = 8080
const path = __dirname + '/public';
const http = require('http');
const socketio = require('socket.io');
const fs = require('fs');
var bodyParser = require('body-parser'); 
var multer  = require('multer');
var process = require('child_process');
var SocketIOFileUpload = require('socketio-file-upload');


app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json())  
app.use(express.static(path))
app.use(SocketIOFileUpload.router);



const server = http.createServer(app);
const io = socketio(server);



io.on('connection', (sock) => {
  console.log('[*] - New user connected - client-' + sock.id);
  var uploader = new SocketIOFileUpload();
  uploader.dir = __dirname+"/uploads";
  uploader.listen(sock);
  uploader.on("saved", function(event){
      console.log(event);
  });
  // Error handler:
  uploader.on("error", function(event){
      console.log("Error from uploader", event);
  });


  sock.on('upload', (data) => {
      fs.readdirSync(__dirname+"/uploads").forEach(file => {
        if(file != ".DS_Store") {
            fs.readFile(__dirname+"/uploads/"+file, (err,buf) => {
                sock.emit('loading_video', { video: true, buffer: buf.toString('base64'), name: file });
            });
        }
    }); 
  });
  sock.on('rm_vid', (data) => {
    console.log(data);
    process.exec('rm ./uploads/' + data, function (err,stdout,stderr) {
        if (err) {
            console.log("\n"+stderr);
        } else {
        console.log(stdout);
        }
    });
  });
  sock.on('run', (data) => {
    fs.readdirSync(__dirname+"/uploads").forEach(file => {
        if(file != ".DS_Store") {
          var pid_ = process.exec('python algorithm.py -v ' + file, function (err,stdout,stderr) {
              if (err) {
                  console.log("\n"+stderr);
              } else {
              console.log(stdout);
              }
          }); 
          pid_.on("close",() => {
              fs.readFile(__dirname + "/output/final_output-"+file, (err,buf) => {
                var payl={
                    video:true,
                    buffer: buf.toString('base64'),
                    name: file,
                }
                sock.emit('output',payl);
              });
          });

        }
    });
  });
  
 

  sock.on('disconnect', (reason) => {
      console.log(sock.id + " disconnected");
  });

});

  



server.on('error', (err) => {
  console.log(err);
});

server.listen(port, () => {
  console.log("[*] - Server attivo...");
});










// app.get('/', (req, res) => res.sendFile(path +"/index.html"));

// app.post('/result', upload.single('video'), function (req, res, next) {
//     // req.file is the `avatar` file
//     // req.body will hold the text fields, if there were any
//     console.log(req.file, req.body)  
//   var pid_ = process.exec('python videotoframe.py',function (err,stdout,stderr) {
//             // res.sendFile(path + "/output.html") //path video
//             });

//   pid_.on("close",() => {
//     res.send({error: false, ouput: "final_output.mp4" })
//   });
//   pid_.on("error",(err) => {
//     res.send({error: true,msg: err})
//   }); 
// })

// app.listen(port, () => console.log(`[*] - Server online --> port ${port}!`))