var song_count = db.getCollection("songs").count();
var time = db.isMaster().localTime;
print("song_count: " + song_count+",time: " + time);
sleep(120000)
var song_count2 = db.getCollection("songs").count();
var time2 = db.isMaster().localTime;
print("song_count: " + song_count2+",time: " + time2);
var time_delta = time2-time;
var song_count_delta=song_count2-song_count;
print("song_count_delta: " + song_count_delta+",time_delta: " + time_delta);
print(song_count_delta/(time_delta/1000)+" songs per second");