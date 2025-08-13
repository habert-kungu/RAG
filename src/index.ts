import express from "express";
const app = express;
const port = 3000;
const hostname = "127.0.0.1";

app.get("/", (req, res) => {
  res.send("hello world");
});

app.listen(port, () => {
  return console.log(`Server listening on https://${hostname}:${port}`);
});
