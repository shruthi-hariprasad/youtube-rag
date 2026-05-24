import { useState, useEffect, useRef } from "react"
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
  summary?: string | null
  suggested_questions?: string | null
}

const PROGRESS_STEPS = [
  "Fetching transcript...",
  "Processing chunks...",
  "Generating embeddings...",
  "Generating summary...",
]

export default function Library() {
  const [videos, setVideos] = useState<Video[]>([])
  const [url, setUrl] = useState("")
  const [loading, setLoading] = useState(false)
  const [fetching, setFetching] = useState(true)
  const [error, setError] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [progressStep, setProgressStep] = useState(0)
  const progressRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const navigate = useNavigate()

  const filteredVideos = videos.filter(v => {
    const q = searchQuery.toLowerCase()
    return (
      v.title?.toLowerCase().includes(q) ||
      v.channel_name?.toLowerCase().includes(q)
    )
  })

  function handleSearchLibrary() {
    navigate("/chat", { state: { video: null } })
  }

  useEffect(() => {
    api.get("/videos")
      .then(res => setVideos(res.data))
      .catch(() => setError("Failed to load videos"))
      .finally(() => setFetching(false))
  }, [])

  async function handleDelete(e: React.MouseEvent, videoId: number) {
    e.stopPropagation()
    if (!window.confirm("Remove this video from your library?")) return
    try {
      await api.delete(`/videos/${videoId}`)
      setVideos(prev => prev.filter(v => v.id !== videoId))
    } catch {
      setError("Failed to delete video")
    }
  }

  async function handleAddVideo(e: React.FormEvent) {
    e.preventDefault()
    setError("")
    setLoading(true)
    setProgressStep(0)
    progressRef.current = setInterval(() => {
      setProgressStep(prev => (prev < PROGRESS_STEPS.length - 1 ? prev + 1 : prev))
    }, 4000)

    try {
      const res = await api.post(`/videos?url=${encodeURIComponent(url)}`)
      setVideos(prev => [res.data, ...prev])
      setUrl("")
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to add video")
    } finally {
      if (progressRef.current) clearInterval(progressRef.current)
      setProgressStep(0)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-4xl mx-auto px-6 py-8">
        <button
          onClick={handleSearchLibrary}
          className="w-full bg-white border border-indigo-200 hover:border-indigo-400 text-indigo-600 font-medium px-5 py-3 rounded-xl text-sm transition mb-4 flex items-center justify-center gap-2"
        >
          Search my entire library
        </button>

        <form onSubmit={handleAddVideo} className="flex gap-3 mb-4">
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
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-lg text-sm transition disabled:opacity-60 whitespace-nowrap min-w-[160px]"
          >
            {loading ? PROGRESS_STEPS[progressStep] : "Add Video"}
          </button>
        </form>

        {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

        {videos.length > 0 && (
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Filter by title or channel..."
            className="w-full border border-gray-200 rounded-lg px-4 py-2.5 text-sm mb-6 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        )}

        {fetching ? (
          <p className="text-gray-400 text-sm">Loading your library...</p>
        ) : videos.length === 0 ? (
          <div className="text-center py-20 text-gray-400">
            <p className="text-lg mb-1">No videos yet</p>
            <p className="text-sm">Paste a YouTube URL above to get started</p>
          </div>
        ) : filteredVideos.length === 0 ? (
          <p className="text-gray-400 text-sm">No videos match "{searchQuery}"</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredVideos.map(video => (
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
                <div className="p-3 flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-gray-900 line-clamp-2">{video.title}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{video.channel_name}</p>
                    {video.summary && (
                      <p className="text-xs text-gray-400 mt-1.5 line-clamp-2 leading-relaxed">{video.summary}</p>
                    )}
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, video.id)}
                    className="shrink-0 text-gray-300 hover:text-red-500 transition text-xl leading-none pt-0.5"
                    title="Remove video"
                  >
                    &times;
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
