import { useState } from "react"
import { useLocation, useNavigate } from "react-router-dom"
import api from "../api/axios"
import Navbar from "../components/Navbar"

interface Source {
  video_id: string
  title: string
  chunk_index: number
  text: string
}

interface Message {
  role: "user" | "assistant"
  content: string
  sources?: Source[]
}

interface Video {
  youtube_video_id: string
  title: string
  channel_name: string
  thumbnail_url: string
  url: string
}

export default function Chat() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const video: Video | null = state?.video ?? null

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage: Message = { role: "user", content: input }
    setMessages(prev => [...prev, userMessage])
    setInput("")
    setLoading(true)
    setError("")

    try {
      const queryUrl = video
        ? `/query?question=${encodeURIComponent(input)}&video_id=${video.youtube_video_id}`
        : `/query?question=${encodeURIComponent(input)}`
      const res = await api.post(queryUrl)
      const assistantMessage: Message = {
        role: "assistant",
        content: res.data.answer,
        sources: res.data.sources,
      }
      setMessages(prev => [...prev, assistantMessage])
    } catch (err: any) {
      setError(err.response?.data?.detail || "Something went wrong")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">

      <Navbar />

      {/* Sub-header: back button + video info */}
      <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-4">
        <button
          onClick={() => navigate("/library")}
          className="text-gray-400 hover:text-gray-600 text-sm"
        >
          ← Library
        </button>

        {video ? (
          <>
            <img
              src={video.thumbnail_url}
              alt={video.title}
              className="w-14 rounded-lg object-cover"
            />
            <div>
              <p className="text-sm font-semibold text-gray-900 line-clamp-1">{video.title}</p>
              <p className="text-xs text-gray-500">{video.channel_name}</p>
            </div>
          </>
        ) : (
          <div>
            <p className="text-sm font-semibold text-gray-900">Search your entire library</p>
            <p className="text-xs text-gray-500">Answers will include which video they came from</p>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 max-w-3xl w-full mx-auto px-6 py-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg mb-1">
              {video ? "Ask anything about this video" : "Ask anything across your library"}
            </p>
            <p className="text-sm">
              {video
                ? "The answer will be grounded in the transcript"
                : "The answer will tell you which video it came from"}
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i}>
            {msg.role === "user" ? (
              <div className="flex justify-end">
                <div className="bg-indigo-600 text-white text-sm px-4 py-3 rounded-2xl rounded-tr-sm max-w-xl">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <div className="bg-white border border-gray-200 text-gray-800 text-sm px-4 py-3 rounded-2xl rounded-tl-sm max-w-xl shadow-sm">
                  {msg.content}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs text-gray-400 font-medium uppercase tracking-wide">Sources</p>
                    {msg.sources.map((src, j) => (
                      <div key={j} className="bg-gray-100 rounded-lg px-4 py-2.5 text-xs text-gray-600 max-w-xl">
                        <p className="font-medium text-gray-700 mb-1">{src.title} · chunk {src.chunk_index}</p>
                        <p className="line-clamp-3 leading-relaxed">{src.text}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex gap-1 px-4 py-3">
            <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:0ms]" />
            <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:150ms]" />
            <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce [animation-delay:300ms]" />
          </div>
        )}

        {error && <p className="text-red-500 text-sm">{error}</p>}
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <form onSubmit={handleAsk} className="max-w-3xl mx-auto flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={video ? "Ask a question about this video..." : "Ask anything across your library..."}
            className="flex-1 border border-gray-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-lg text-sm transition disabled:opacity-50"
          >
            Ask
          </button>
        </form>
      </div>

    </div>
  )
}