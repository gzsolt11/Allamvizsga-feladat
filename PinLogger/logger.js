var outputList = [] // [ [number, pressing timestamp, releasing timestamp],... ]
var pressedNumber = [];
var pressingTimestamp = [];
var releasingTimestamp = [];
var fingerarea = [];
var keydownMutex = false;
var keyupMutex = true;
var size = 0;
var counter = 20;

// Adding input listener to pin input box
var pinAnswer = document.getElementById("pinAnswer");
pinAnswer.addEventListener("input",textChangeListener);

// Generating random PIN and putting it to the screen
var randomPin = createRandomNumber();
document.getElementById("pincode").innerHTML = randomPin;
var pinCode = document.getElementById("pincode").innerHTML.toString();

// Putting the counter to the screen
document.getElementById("counter").innerHTML = counter;

// Adding touchstart and touchend listeners to the buttons
var buttons = document.getElementsByClassName("button");
for(var i = 0; i < buttons.length; ++i){
    buttons[i].addEventListener("touchstart",keydown);
    buttons[i].addEventListener("touchend",keyup);
}

// Adding click listener to download button
var downloadButton = document.getElementById("downloadButton");
downloadButton.addEventListener("click",downloadFile);

// Adding click listener to the copy email button
document.getElementById("copyImage").addEventListener("click",copyEmail);

// Random 6 non-repeating digits PIN generator function 
function createRandomNumber(){
    var number = 0;
    var digits = [];
    var multiplier = 1;
    for(var i = 0; i < 6; ++i){
        if(i == 1){
            digits.push(0);
        }
        do{
            var temp = Math.floor(Math.random() * 10);
        }while(digits.includes(temp));
        digits.push(temp);
        number = number + (temp * multiplier);
        multiplier *= 10;
    }
    return number;
}

// Checks if the user typed a good or a wrong character, deletes the input field value if failed or typed the whole PIN correct
// Add the touch event properties into the output list
function textChangeListener(e){
    var length = (pinAnswer.value.toString()).length;
    if(counter > 0){
        if(pinCode.substr(0,length) !== pinAnswer.value.toString() ){
            pinAnswer.value = "";
            pinAnswer.classList.add("fail");
            resetLists();
        }else if(length == 6){
            pinAnswer.value = "";
            if(counter > 0){
                counter -= 1;
                if(counter == 0){
                    popUpWindow();
                }
            }
            document.getElementById("counter").innerHTML = counter;
            addListsToOutput();
            resetLists();
            pinAnswer.classList.add("success");
    
        }
    }else{
        pinAnswer.value = "";
    }
    
}

// At pressing a button the touchevent properties will be added to certain lists
function keydown(event){
    if(!keydownMutex){
        keydownMutex = true;
        keyupMutex = false;
        pressingTimestamp.push(Date.now());
        pressedNumber.push(event.target.innerHTML);
    }
}

// At releasing the button the release timestamp will be added to it's list and the number will be added to the input field
function keyup(event){
    if(!keyupMutex){
        keyupMutex = true;
        keydownMutex = false;
        releasingTimestamp.push(Date.now());
        writeNumber(event.target.innerHTML);
    }
}

// Add the number to the input field and triggers input event
function writeNumber(number){
    pinAnswer.classList.remove("success")
    pinAnswer.classList.remove("fail")
    pinAnswer.value = pinAnswer.value +number;

    var event = new Event('input',{
        bubbles:true,
        cancelable:true,});
    pinAnswer.dispatchEvent(event);
}

// reseting all lists with empty lists
function resetLists(){
    pressingTimestamp = [];
    releasingTimestamp = [];
    pressedNumber = [];
}

// Merge the lists into a big list 
function addListsToOutput(){
    for(var i = 0; i < pressedNumber.length; ++i){
        outputList.push([parseInt(pressedNumber[i]),pressingTimestamp[i],releasingTimestamp[i]]);
    }
}

// Converts our output list into a csv list
function convertListToCSV(list){
    var CSVRows = ["key,pressTime,releaseTime"];
    for(var i = 0; i < list.length; ++i){
        var line = list[i].join(",");
        CSVRows.push(line);
    }
    var CSVData = CSVRows.join('\n');
    return CSVData;
}

// Downloads the file to our device
function downloadFile(){
    var outputCSV = convertListToCSV(outputList);
    const blob = new Blob([outputCSV], { type: 'text/csv'});
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden','');
    a.setAttribute('href',url);
    a.setAttribute('download','keylog.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Copies the e-mail adress from myEmail element
function copyEmail(){
    var x = document.getElementById("toastMessage");
    x.className = "show";
    setTimeout(function(){ x.className = x.className.replace("show", ""); }, 3000);
    var copyText = document.getElementById("myEmail");
    copyText.select();
    copyText.setSelectionRange(0, 99999);
    document.execCommand("copy");
}

// Show a toast message to use mobile device
function askForMobile(){
    var x = document.getElementById("toastMessageForMobile");
    x.className = "show";
    setTimeout(function(){ x.className = x.className.replace("show", ""); }, 3000);
}

// Shows the popUp window if all the PINs were typed correctly and blocks the inputscreen
function popUpWindow(){
    var popUpBlock = document.getElementById("popUpBlock");
    document.getElementById("inputScreen").style['poiinter-event'] = "none";
    popUpBlock.style.visibility = "visible";
    popUpBlock.className = "blockPopUp";
    setTimeout(function(){ popUpBlock.className = popUpBlock.className.replace("blockPopUp", ""); }, 1000);
}