const sock = io.connect("http://localhost:8080");
var run = document.getElementById('run'); 
var btn_upload_video = document.getElementById('upload'); 
var dataset = document.getElementById('dataset');
var dataset_output = document.getElementById('output');
var hashkey_input = [];
var videos_output = [];
var videos_count = 0;
var videos_output_count = 0;
var onces = true;


btn_upload_video.addEventListener('click', () => {
    sock.emit("upload","");
    var elem = document.querySelector('#dataset');
    elem.style.border = '3px solid #2e7a47';
});

run.addEventListener('click', () => {
    sock.emit("run","");
});

const zoom_video = (data) => {
    zoom_output = document.getElementById('zoom_output');
    var elem = '<br><br><div style="text-align:center" ><h3 class="upper-case">'+data+'</h3>\
            <video class="ml-3 mb-5" width="80%" controls>\
                <source src="data:video/mp4;base64,'+videos_output[data]+'" type="video/mp4">\
            </video><br><br>';
    zoom_output.innerHTML = elem;
}

const mydelete_video = (index) => {
    var exit= document.getElementById('exit-'+hashkey_input[index]);
    var elim= document.getElementById("video-"+hashkey_input[index]);
    sock.emit('rm_vid', exit.alt);
    elim.parentNode.removeChild(elim);
}
const print_set_video = (data) => {
    var elem ='';
    elem+='<li id="video-'+data.name+'"><video class="ml-3 item" width="400"  controls>\
            <source id="'+data.name+'" src="data:video/mp4;base64,'+data.buffer+'" type="video/mp4">\
            </video>\
            <img id="exit-'+data.name+'" src="error.png" width="20px" onclick="mydelete_video('+videos_count+')" class="show_vid" alt="'+data.name+'">\
            <span class="show_vid_text">'+data.name+'</span></li>';    
    hashkey_input[videos_count] = data.name;
    console.log(hashkey_input[videos_count])
    videos_count+=1;
    dataset.innerHTML+=elem;
}
const print_output_video = (video,text,cam) => {
    var elem = document.querySelector('#output');
    elem.style.border = '3px solid #2e7a47';
    var elem = '<li id="video-'+video.name+'"><video class="ml-3 item" width="400" onclick="zoom_video('+videos_output_count+')">\
                    <source id="'+video.name+'" src="data:video/mp4;base64,'+video.buffer+'" type="video/mp4">\
                    </video>\
                    <span class="show_vid_text">'+video.name+' - '+videos_output_count+'</span></li>';    
    videos_output[videos_output_count] = video.buffer;
    videos_output_count+=1;        
    dataset_output.innerHTML+=elem;

    
}

const fun_output = () => {
    onces = true;
}


sock.on('loading_video', (data) => {
    print_set_video(data);
});
sock.on('output', (data) => {
    try {
    console.log(data);
    var elim= document.getElementById("dataset");
    elim.parentNode.removeChild(elim);
    var elim2= document.getElementById("upl");
    elim2.parentNode.removeChild(elim2);
    var elim3= document.getElementById("text-up");
    elim3.parentNode.removeChild(elim3);
    var elim4= document.getElementById("run");
    elim4.parentNode.removeChild(elim4);
    modal_body = document.getElementById('loading');
    modal_body.parentNode.removeChild(modal_body);
    modal_body_gif = document.getElementById('loadgif');
    modal_body_gif.parentNode.removeChild(modal_body_gif);
    var parent = document.body;
    var brs = parent.getElementsByTagName('br');
    for(var i = brs.length-3; i--;) {
        brs[i].parentNode.removeChild(brs[i]);
    }
    }
    catch(err) {
        console.log(err.message);
    }
    if(onces==true) {
    content = document.querySelector('.modal-body');
    title = document.querySelector('#title_output');
    foot = document.querySelector('.modal-footer');
    var elem = '<h1>Esecuzione completata!</h1><p>Clicca sul pulsante qui in basso.</p>';
    var elem2 = '<button type="button" class="btn btn-default center" data-dismiss="modal" onclick="fun_output()">Output</button>';
    content.innerHTML+=elem;
    foot.innerHTML+=elem2;
    onces=false;
    title.style.display="block";
    }
    print_output_video(data); 
});

