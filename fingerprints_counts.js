// Stages that have been excluded from the aggregation pipeline query
__3tsoftwarelabs_disabled_aggregation_stages = [

	{
		// Stage 4 - excluded
		stage: 4,  source: {
			$project: {
			    // specifications
			    count:1,
			    length:"$song.length"
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

		// Stage 2
		{
			$sort: {
			count:-1
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

	]

	// Created with Studio 3T, the IDE for MongoDB - https://studio3t.com/

);
