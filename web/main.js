function clusterFaces() {
    clearDisplayFaces()
    clearErrorMessage()
	var filePath = document.getElementById("filePath").value
//	console.log(filePath, document.getElementById("filePath").files[0].path)
	var secondsPerFrame = document.getElementById("secondsPerFrame").value
    eel.detect_faces_from_video(filePath, parseFloat(secondsPerFrame) )(display_faces)
}

function addText(text) {
    var div = document.getElementById("progress");
    div.innerHTML = text;
}
eel.expose(addText);

function addErrorMessage(text) {
    var div = document.getElementById("errorDiv");
    div.innerHTML = "<strong>ApplicationError!</strong>"+text;
    div.classList.add("alert-danger")
}
eel.expose(addErrorMessage);

function clearErrorMessage(){
    var div = document.getElementById("errorDiv");
    div.innerHTML ="";
    div.classList.remove("alert-danger")
}

function display_faces(detected_faces){
    var div = document.getElementById("progress");
    div.innerHTML = "";
    var container = document.getElementById("display-faces")
    for (var key in detected_faces){
        var mainRow = createDiv("row", "", "display-faces-row")
        var labelColumn = createDiv("col-2 d-flex justify-content-center text-center", "<h6>"+key+"</h6>")
        mainRow.appendChild(labelColumn);
        var faceDivs = createFaceDivs(detected_faces[key])
        var faceColumn = createFaceColumn(faceDivs)
        mainRow.appendChild(faceColumn);
        container.appendChild(mainRow)
    }

}

function clearDisplayFaces(){
    var div = document.getElementById("display-faces")
    div.innerHTML = "";
}

function createDiv(className, innerHTML="", id=""){
    var innerDiv = document.createElement('div');
    innerDiv.className = className;
    innerDiv.innerHTML = innerHTML
    innerDiv.id = id
    return innerDiv
}

function createFaceDivs(detected_faces_path){
    var faceDivs = []
    for (var index in detected_faces_path ){
        var img = "<img src='"+detected_faces_path[index]+"' class='rounded mx-auto d-block img-responsive' style='width:40px;height:40px;'>"
        faceDivs.push(createDiv("col-2", img))
    }
    return faceDivs
}

function createFaceColumn(faceDivs){
    var faceColumn = createDiv("col-10")
    for (var i=0; i <= faceDivs.length ; i+=5){
        var faceRow = createDiv("row", "", ""+i)

        var maxElement = i+5
        if (maxElement > faceDivs.length){
            maxElement = faceDivs.length
            }
        var slicedDivs = faceDivs.slice(i,maxElement)
        for(var index in slicedDivs){
            console.log(i, index, slicedDivs[index])
            faceRow.appendChild(slicedDivs[index])
        }
        faceColumn.appendChild(faceRow)
    }
    return faceColumn
}