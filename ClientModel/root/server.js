var express = require('express');
var app = express();

app.use(express.static(__dirname));

app.listen(8000, function() {
  console.log('Examples server running at http://localhost:8000 (http://10.176.37.50:8000)');
});
