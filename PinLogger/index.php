<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0" >
    <title>Document</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>

<?php 

function isMobileDevice() { 
    return preg_match("/(android|avantgo|blackberry|bolt|boost|cricket|docomo|fone|hiptop|mini|mobi|palm|phone|pie|tablet|up\.browser|up\.link|webos|wos)/i" , $_SERVER["HTTP_USER_AGENT"]); 
} 
if(isMobileDevice()){ ?>
    <div id="inputScreen">
        <div id="titleBlock">
            <div id="title">A te PIN kódod: </div>
            <div id="pincode"></div>
            <div class="request">Kérlek gépeld be a kódot!</div>
            <div class="request">Csak a helyes gépelések mentődnek.</div>
        </div>
        
        <div id="counterText">Még <div id="counter"></div> alkalommal kell helyesen beütnöd a kódot</div>
        <input id="pinAnswer" type="number">
        <div id="buttons">
            <div class="button" id="button1">1</div>
            <div class="button" id="button2">2</div>
            <div class="button" id="button3">3</div>
            <div class="button" id="button4">4</div>
            <div class="button" id="button5">5</div>
            <div class="button" id="button6">6</div>
            <div class="button" id="button7">7</div>
            <div class="button" id="button8">8</div>
            <div class="button" id="button9">9</div>
            <div class="button" id="button0">0</div>
        </div>
        
        <div class="sourceCode">Az adatokat névtelenül mentjük és dolgozzuk fel!</div>
        <div class="sourceCode">Forráskód: <a id="sourceCodeLink" href="https://github.com/gzsolt11/PinLogger.git">https://github.com/gzsolt11/PinLogger.git</a></div>
        
        <div id="toastMessageForMobile">Kérlek mobil eszköz segítségével végezd a mérést</div>
    </div>


    <div id="popUpBlock" >
        <div id="popUpWindow">
            <div id="popUpBody">
                <div id="popUpTitle">
                    Köszönjük szépen, hogy résztvettél a mérésen!
                </div>
                <div id="popUpInstruction">
                    A letöltés gomb megnyomásával letöltheted a mérés eredményeit, majd kérünk küld el a következő e-mail címre: 
                </div>
                <div id="emailAdress">manyi@ms.sapientia.ro</div>
                <div id="copyEmail">E-mail cím másolása:<div id="copyImage"><img src="./src/copy-icon.png" alt=""></div></div>
                <button id="downloadButton">Letöltés</button>
            </div>
            <div id="toastMessage">E-mail cím másolva vágólapra</div>
        </div>
        
        <input id="myEmail" value="manyi@ms.sapientia.ro" readonly>
    </div>


    
    <script src="logger.js" defer type="module"></script>


<?php
}
else { 
    echo "Kérlek az oldalt mobil eszköz segítségével nyítsd meg"; 
} 
?> 
    

    

</body>
</html>