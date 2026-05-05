import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import api from "../api/axios"
import Navbar from "../components/Navbar"

interface Video {
  id: number
  youtube_video_id: string
  title: string
  channel_name: string
  thumbnail_url: string
  url: string
}

export default function Library() {
  const [videos, setVideos] = useState<Video[]>([])
  const [url, setUrl] = useState("")
  const [loading, setLoading] = useState(false)
  const [fetching, setFetching] = useState(true)
  const [error, setError] = useState("")
  const navigate = useNavigate()

  function handleSearchLibrary() {
  navigate("/chat", { state: { video: null } })
}

  useEffect(() => {
    api.get("/videos")
      .then(res => setVideos(res.data))
      .catch(() => setError("Failed to load videos"))
      .finally(() => setFetching(false))
  }, [])

  async function handleAddVideo(e: React.FormEvent) {
    e.preventDefault()
    setError("")
    setLoading(true)
    try {
      const res = await api.post(`/videos?url=${encodeURIComponent(url)}`)
      setVideos(prev => [res.data, ...prev])
      setUrl("")
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to add video")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Search library button */}
        <button
          onClick={handleSearchLibrary}
          className="w-full bg-white border border-indigo-200 hover:border-indigo-400 text-indigo-600 font-medium px-5 py-3 rounded-xl text-sm transition mb-4 flex items-center justify-center gap-2"
        >
          🔍 Search my entire library
        </button>

        {/* Add video form */}
        <form onSubmit={handleAddVideo} className="flex gap-3 mb-8">
          <input
            type="text"
            value={url}
            onChange={e => setUrl(e.target.value)}
            placeholder="Paste a YouTube URL..."
            className="flex-1 border border-gray-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            required
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-lg text-sm transition disabled:opacity-50 whitespace-nowrap"
          >
            {loading ? "Adding..." : "Add Video"}
          </button>
        </form>

        {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

        {/* Video grid */}
        {fetching ? (
          <p className="text-gray-400 text-sm">Loading your library...</p>
        ) : videos.length === 0 ? (
          <div className="text-center py-20 text-gray-400">
            <p className="text-lg mb-1">No videos yet</p>
            <p className="text-sm">Paste a YouTube URL above to get started</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {videos.map(video => (
              <div
                key={video.id}
                onClick={() => navigate("/chat", { state: { video } })}
                className="bg-white rounded-xl overflow-hidden shadow-sm hover:shadow-md transition cursor-pointer"
              >
                <img
                  src={video.thumbnail_url}
                  alt={video.title}
                  className="w-full aspect-video object-cover"
                />
                <div className="p-3">
                  <p className="text-sm font-medium text-gray-900 line-clamp-2">{video.title}</p>
                  <p className="text-xs text-gray-500 mt-1">{video.channel_name}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}