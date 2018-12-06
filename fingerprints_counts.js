// Stages that have been excluded from the aggregation pipeline query
__3tsoftwarelabs_disabled_aggregation_stages = [

	{
		// Stage 2 - excluded
		stage: 2,  source: {
			$sort: {
			count:-1
			}
		}
	},

	{
		// Stage 8 - excluded
		stage: 8,  source: {
			$sort: {
			fingerprintsPerSecond:-1
			}
		}
	},
]

db.getCollection("fingerprints").aggregate(

	// Pipeline
	[
		// Stage 1
		{
			$group: {
			_id :"$songID",
			count: { $sum: 1 }
			}
		},

		// Stage 3
		{
			$lookup: // Equality Match
			{
			    from: "songs",
			    localField: "_id",
			    foreignField: "_id",
			    as: "song"
			}
		},

		// Stage 4
		{
			$unwind: {
			    path : "$song",
			}
		},

		// Stage 5
		{
			$addFields: {
			    songLength:"$song.track_length_s"
			}
		},

		// Stage 6
		{
			$project: {
			    song:0
			}
		},

		// Stage 7
		{
			$addFields: {
			    "fingerprintsPerSecond": {"$divide":["$count","$songLength"]}
			}
		},

		// Stage 9
		{
			$group: {
			"_id": null, 
			"avgCount": { "$avg": "$count" },
			"avgSongLength": { "$avg": "$songLength" },
			"avgFPpS": { "$avg": "$fingerprintsPerSecond" } 
			}
		},

	]

	// Created with Studio 3T, the IDE for MongoDB - https://studio3t.com/

);
